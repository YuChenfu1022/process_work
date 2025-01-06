import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,4'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,4,5,6'
import pickle
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args, args_gnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = list(range(torch.cuda.device_count()))  
print(f"using device {device}")
# print(device_ids)

# dist.init_process_group(backend='nccl')
# local_rank = dist.get_rank()
# # print(f'local_rank: {local_rank}')
# device_id = local_rank % torch.cuda.device_count()
# torch.cuda.set_device(device_id)
# dist.barrier()
# print(f"Rank {local_rank} is using device {device_id}")

class XRec:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        self.model = Explainer().to(device)
        # self.model = Explainer().to(device_id)
        self.data_handler = DataHandler()
        # self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()
        self.user_embedding_converter_path = f"./data/{args.dataset}/user_converter.pkl"
        self.item_embedding_converter_path = f"./data/{args.dataset}/item_converter.pkl"
        self.tst_predictions_path = f"./data/{args.dataset}/tst_predictions.pkl"
        self.tst_references_path = f"./data/{args.dataset}/tst_references.pkl"

    def train(self):
        # self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam([
        #                 {'params': self.model.user_embedding_converter.parameters(), 'lr': args.lr,}, 
        #                 {'params': self.model.item_embedding_converter.parameters(), 'lr': args.lr,},
        #                 {'params': self.model.gnn_model.lightgcn_model.parameters(), 'lr': args_gnn.lr},
        #             ])

        self.trn_loader, _, _ = self.data_handler.load_data()
        # self.trn_dataset, _, _ = self.data_handler.load_data()
        # train_sampler = torch.utils.data.distributed.DistributedSampler(self.trn_dataset)
        # self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)

        for epoch in range(args.epochs):
            total_loss = 0

            self.model.gnn_model.train()
            self.data_handler.load_emb()

            self.model.train()
            torch.cuda.empty_cache()  # 释放显存
            for i, batch in enumerate(tqdm(self.trn_loader)):
                # torch.cuda.empty_cache()  # 释放显存
                # if i == 250:
                #     break
                user_embed, item_embed, input_text = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)
                input_ids, outputs, explain_pos_position = self.model.forward(user_embed, item_embed, input_text)
                input_ids = input_ids.to(device)
                explain_pos_position = explain_pos_position.to(device)
                optimizer.zero_grad()
                loss = self.model.loss(input_ids, outputs, explain_pos_position, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # if (i + 1) % 50 == 0 and i != 0:
                #     print(
                #         f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(self.trn_loader)}], Loss: {loss.item()}"
                #     )
                #     print(f"Generated Explanation: {outputs[0]}")

            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss}")
            # Save the model
            self.data_handler.save_emb()
            torch.save(
                self.model.user_embedding_converter.state_dict(),
                self.user_embedding_converter_path,
            )
            torch.save(
                self.model.item_embedding_converter.state_dict(),
                self.item_embedding_converter_path,
            )
            print(f"Saved model to {self.user_embedding_converter_path}")
            print(f"Saved model to {self.item_embedding_converter_path}")

    def evaluate(self):
        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()
        loader = self.tst_loader
        predictions_path = self.tst_predictions_path
        references_path = self.tst_references_path

        # load model
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device) # size: [batch_size, 64]
                item_embed = item_embed.to(device)
                
                outputs = self.model.generate(user_embed, item_embed, input_text)
                end_idx = outputs[0].find("[") 
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                # predictions.append(outputs[0])
                # references.append(explain[0])
                # print(f"step_{i}, input_text: {input_text}")
                for o, e in zip(outputs, explain):
                    predictions.append(o)
                    references.append(e)
                    print(f"outputs: {o}")
                    print(f"explain: {e}")
                
                if i % 50 == 0 and i != 0:
                    print(f"Step [{i}/{len(loader)}]")
                    # print(f"Generated Explanation: {outputs[0]}")

        with open(predictions_path, "wb") as file:
            pickle.dump(predictions, file)
        with open(references_path, "wb") as file:
            pickle.dump(references, file)
        print(f"Saved predictions to {predictions_path}")
        print(f"Saved references to {references_path}")   

def main():
    sample = XRec()
    if args.mode == "finetune":
        print("Finetune model...")
        sample.train()
    elif args.mode == "generate":
        print("Generating explanations...")
        sample.evaluate()

if __name__ == "__main__":
    main()
