import os
import json

import torch
from torch.utils.data import DataLoader, Subset

from muscall.models.muscall import MusCALL
from muscall.datasets.audiocaption import AudioCaptionDataset

import numpy as np

@torch.no_grad()
def get_muscall_features(model, data_loader, device):
    dataset_size = data_loader.dataset.__len__()

    all_audio_features = torch.zeros(dataset_size, 512).to(device)
    all_text_features = torch.zeros(dataset_size, 512).to(device)

    samples_in_previous_batch = 0

    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        _, input_audio, text_input_ids, _, _, _ = batch

        audio_features = model.encode_audio(input_audio)
        text_features = model.encode_text(text_input_ids, None)

        audio_features = audio_features / \
            audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        samples_in_current_batch = input_audio.size(0)
        start_index = i * samples_in_previous_batch
        end_index = start_index + samples_in_current_batch
        samples_in_previous_batch = samples_in_current_batch

        all_audio_features[start_index:end_index] = audio_features
        all_text_features[start_index:end_index] = text_features

    return all_audio_features, all_text_features

'''
Compute a similarity score matrix by multiplying the audio features matrix with the transpose of the text feature matrix

logits_per_audio: each row is an audio, each column is a text
logits_per_text: each row is a text, each column is an audio

In logits_per_text, each entry at row i, column j represents the similary score
between the audio feature vector of and text i and audio j.
'''
def compute_sim_score(audio_features, text_features):
    logits_per_audio = audio_features @ text_features.t() # matrix multiplication + transpose
    logits_per_text = logits_per_audio.t()

    return logits_per_text


def get_ranking(score_matrix, device):
    num_queries = score_matrix.size(0) # queries = texts
    num_items = score_matrix.size(1) # items = audio files

    scores_sorted, retrieved_indices = torch.sort(
        score_matrix, dim=1, descending=True)
    
    # num_queries: number of 2D matrix
    # num_items: number of rows in each 2D matrix
    # 1: number of columns in each 2D matrix
    gt_indices = torch.zeros((num_queries, num_items, 1))

    # print("score_matrix", score_matrix) 
    # print("scores_sorted", scores_sorted)
    # print("retrieved_indices: ", retrieved_indices)

    for i in range(num_queries):
        gt_indices[i] = torch.full((num_queries, 1), i)
    # gt_indices = tensor([[[0.],
                    #      [0.],
                    #      [0.]],

                    #     [[1.],
                    #      [1.],
                    #      [1.]],

                    #     [[2.],
                    #      [2.],
                    #      [2.]],
                    # ])

    gt_indices = gt_indices.squeeze(-1).to(device)
    # gt_indices = tensor([[0., 0., 0., 0., 0.],
                        # [1., 1., 1., 1., 1.],
                        # [2., 2., 2., 2., 2.],
                        # [3., 3., 3., 3., 3.],
                        # ...
                        # []])
    return retrieved_indices, gt_indices


def compute_metrics(retrieved_indices, gt_indices, eval=False):
    num_items = gt_indices.size(1)

    bool_matrix = retrieved_indices == gt_indices

    r1 = 100 * bool_matrix[:, 0].sum() / num_items # bool_matrix[: 0] selects all rows of the first column (column with index 0)
    r5 = 100 * bool_matrix[:, :5].sum() / num_items # bool_matrix[:, 5] selects all rows of the first 5 columns (exclude column index 5)
    r10 = 100 * bool_matrix[:, :10].sum() / num_items

    median_rank = (torch.where(bool_matrix == True)[1] + 1).median()

    retrieval_metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "Median Rank": median_rank,
    }


    retrieval_metrics_eval = {
        "R@1": r1.item(),
        "R@5": r5.item(),
        "R@10": r10.item(),
        "Median Rank": median_rank.item(),
    }

    if (not eval):
        return retrieval_metrics
    else:
        return retrieval_metrics_eval


def run_retrieval(model, data_loader, device):
    """Wrapper function to run all steps for text-audio/audio-text retrieval"""
    audio_features, text_features = get_muscall_features(
        model, data_loader, device)
    score_matrix = compute_sim_score(audio_features, text_features)
    retrieved_indices, gt_indices = get_ranking(score_matrix, device)
    retrieval_metrics = compute_metrics(retrieved_indices, gt_indices)

    return retrieval_metrics


class Retrieval:
    def __init__(self, muscall_config, test_set_size=0):
        super().__init__()
        self.muscall_config = muscall_config
        self.device = torch.device(self.muscall_config.training.device)
        self.path_to_model = os.path.join(
            self.muscall_config.env.experiments_dir,
            self.muscall_config.env.experiment_id,
            "best_model.pth.tar",
        )
        print("path to model", self.path_to_model)

        self.test_set_size = test_set_size

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        dataset = AudioCaptionDataset(self.muscall_config.dataset_config, dataset_type="test")
        indices = torch.randperm(len(dataset))[: self.test_set_size]
        random_dataset = Subset(dataset, indices)
        # self.batch_size = 256     # 8.9109
        self.batch_size = 10      # 10.8911
        # self.batch_size = 20      # 10.8911
        # self.batch_size = 8         # 11.8812
        # self.batch_size = 16         # 

        self.data_loader = DataLoader(
            dataset=random_dataset,
            batch_size=self.batch_size,
            drop_last=False,
        )

    def build_model(self):
        self.model = MusCALL(self.muscall_config.model_config)
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        # self.evaluate()

    def evaluate(self, output_file):
        audio_features, text_features = get_muscall_features(
            self.model, self.data_loader, self.device
        )
        score_matrix = compute_sim_score(text_features, audio_features)
        retrieved_indices, gt_indices = get_ranking(score_matrix, self.device)
        retrieval_metrics = compute_metrics(retrieved_indices, gt_indices, True)

        json_retrieval_metrics = json.dumps(retrieval_metrics)
        with open(output_file, 'w') as f:
            f.write(json_retrieval_metrics + '\n\n')
        
        # print the indices to a .txt file
        retrieved_indices = retrieved_indices.cpu()

        with open(output_file, 'ab') as f:
            np.savetxt(f, retrieved_indices, '%d')

        return json_retrieval_metrics

