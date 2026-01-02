#!/usr/bin/env python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils import load_adapter, map_adapter_keys

@dataclass
class ModelConfig:
    num_labels: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

class LlamaMortalityClassificationModel(nn.Module):
    """LLaMA-based model for mortality classification tasks."""
    def __init__(self, args, 
        base_model: PreTrainedModel,
    ):
        """
        Initialize the model.
        
        Args:
            base_model: Pre-trained Multi-modal LLaMA model
            use_image: Whether to use image inputs
            use_text: Whether to use text inputs
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.custom_config = ModelConfig
        self.args = args
        
        # Determine hidden size
        self.hidden_size = (
            self.base_model.config.hidden_size
            if hasattr(self.base_model.config, "hidden_size")
            else self.base_model.get_input_embeddings().weight.shape[1]
        )
        
        # Initialize classifier with same dtype as base model
        if not self.args.zeroshot:
            base_dtype = next(self.base_model.parameters()).dtype
            self.classifier = nn.Linear(self.hidden_size, 2).to(base_dtype)     
            self.dropout = nn.Dropout(p=self.custom_config.lora_dropout)  
            self.loss_fn = nn.CrossEntropyLoss()
            self._setup_lora()
    
    def _setup_lora(self) -> None:
        """Configure and apply LoRA adaptation."""
        lora_config = LoraConfig(
            r=self.custom_config.lora_r,
            lora_alpha=self.custom_config.lora_alpha,
            lora_dropout=self.custom_config.lora_dropout,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            init_lora_weights="gaussian",
            bias="none",
            task_type="SEQ_CLS",
        )
        
        self.base_model.language_model.add_adapter(lora_config, adapter_name="language_model_adapter")
        self.base_model.language_model.set_adapter("language_model_adapter")

        if not self.args.use_cxr_image:
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False
            for param in self.base_model.multi_modal_projector.parameters():
                param.requires_grad = False
        else:
            if self.args.lora_setting == 1:
                print("Freeze vision model")
                for param in self.base_model.vision_model.parameters():
                    param.requires_grad = False
            elif self.args.lora_setting == 2:
                print("Train vision model with LoRA")
                self.base_model.vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
                self.base_model.vision_model.set_adapter("vision_model_adapter")
            elif self.args.lora_setting == 3:
                print("Train vision model")
                for param in self.base_model.vision_model.parameters():
                    param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Enable gradient checkpointing if supported."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)
        else:
            print("Warning: base_model does not support gradient checkpointing.")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing logits and optional loss
        """
        # Remove unnecessary kwargs
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("ids", None)
        kwargs.pop("summary_type", None)
        
        if not self.args.use_cxr_image:
            kwargs.pop("pixel_values", None)
        
        if not self.args.zeroshot:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs)

            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            logits = self.classifier(self.dropout(last_hidden_state))
            output_dict = {"logits": logits}

            if labels is not None:
                output_dict["loss"] = self.loss_fn(logits, labels)
                
            return output_dict   
        
        if self.args.summarize:     
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.1,
                **kwargs
            )
            
            return outputs[0][len(input_ids[0]):]  

        if self.args.zeroshot:     
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                **kwargs
            )
            
            return outputs[0][len(input_ids[0]):]            
    
def load_model(
        args,
        model_id : str,
        inference=False
    ) -> Tuple[LlamaMortalityClassificationModel, Union[PreTrainedTokenizer, AutoProcessor]]:
        """
        Load a fine-tuned model and its processor/tokenizer.
        
        Args:
            model_id: Hugging Face model identifier
            checkpoint_dir: Directory containing fine-tuned weights
            use_image: Whether to use image inputs
        
        Returns:
            Tuple of (model, processor/tokenizer)
        """
        processor = AutoProcessor.from_pretrained(model_id)
        if not args.zeroshot:
            processor.tokenizer.padding_side = 'left'
        processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        base_model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

        model = LlamaMortalityClassificationModel(args, base_model)
        if inference:
            checkpoint_path = Path([entry.path for entry in os.scandir(args.checkpoint_dir) if entry.is_dir()][0])
            print(f"\nLoading model from {checkpoint_path}!!")

            classifier_state = torch.load(checkpoint_path / "classifier.bin", map_location="cpu", weights_only=True)
            model.classifier.load_state_dict(classifier_state, strict=False)
            print(f"Loaded classifier...")

            lm_adapter_state = map_adapter_keys(torch.load(checkpoint_path / "lm_adapter.bin", map_location="cpu", weights_only=False), "language_model_adapter")
            current_state_dict = model.base_model.language_model.state_dict()
            load_adapter(current_state_dict, lm_adapter_state)
            model.base_model.language_model.load_state_dict(current_state_dict, strict=False)
            print(f"Loaded language model LoRA adapter...")

            if args.use_cxr_image:
                if args.lora_setting == 1:
                    pass
                elif args.lora_setting == 2:
                    vm_adapter_state = map_adapter_keys(torch.load(checkpoint_path / "vm_adapter.bin", map_location="cpu", weights_only=False), "vision_model_adapter")
                    current_state_dict = model.base_model.vision_model.state_dict()
                    load_adapter(current_state_dict, vm_adapter_state)
                    model.base_model.vision_model.load_state_dict(current_state_dict, strict=False)
                    print(f"Loaded vision model LoRA adapter...")
                elif args.lora_setting == 3:
                    vision_encoder_state = torch.load(checkpoint_path / "vision_encoder.bin", map_location="cpu", weights_only=False)
                    model.base_model.vision_model.load_state_dict(vision_encoder_state, strict=False)
                    print(f"Loaded vision encoder...")
                else:
                    print("Wrong lora setting!")

                multi_modal_projector_state = torch.load(checkpoint_path / "multi_modal_projector.bin", map_location="cpu", weights_only=True)
                model.base_model.multi_modal_projector.load_state_dict(multi_modal_projector_state, strict=False)
                print(f"Loaded multimodal projector...")
            
        return model, processor