import os
import pathlib
import sys

from datasets import load_dataset
from unsloth import FastModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from huggingface_hub import HfApi, upload_folder, create_repo

from pipeline.hf_sync import list_latest_checkpoint, download_folder_prefix
from pipeline.util import load_conf

def _require_token() -> HfApi:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN is required for training uploads")
    return HfApi(token=tok)

def _push_folder(repo_id: str, local_path: str) -> None:
    api = _require_token()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=api.token)
    upload_folder(
        path=local_path,
        repo_id=repo_id,
        repo_type="model",
        token=api.token,
        ignore_patterns=["*.bin", "*.pt", ".git/*"],
    )

def train():
    # Hard fail if CUDA is not available
    try:
        import torch
    except Exception as e:
        print(f"[fatal] PyTorch import failed: {e}", file=sys.stderr); sys.exit(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    cfg = load_conf()
    run_id = cfg["run_id"]
    repos = cfg["repos_fmt"]

    data_jsonl = f"runs/{run_id}/sft/sft_qwenA_from_B_mini.jsonl"
    out_dir = f"runs/{run_id}/trainA"
    os.makedirs(out_dir, exist_ok=True)

    # Resume from latest remote checkpoint if local dir is fresh
    latest = list_latest_checkpoint(repos["a_ckpt_model"])
    if latest and not any(p.name.startswith("checkpoint-") for p in pathlib.Path(out_dir).glob("checkpoint-*")):
        download_folder_prefix(repos["a_ckpt_model"], latest, out_dir)

    ds = load_dataset("json", data_files=data_jsonl, split="train")

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg["model_a_base"],
        load_in_4bit=True,
        max_seq_length=cfg["train"]["max_len"],
        dtype=None,
        device_map="auto",
    )
    model = FastModel.get_peft_model(
        model, r=1, lora_alpha=8, lora_dropout=0.0,
        target_modules="all-linear", bias="none",
        use_gradient_checkpointing="unsloth",
    )

    def fmt(ex): return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg["train"]["bsz"],
        gradient_accumulation_steps=cfg["train"]["grad_acc"],
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type="cosine",
        num_train_epochs=cfg["train"]["epochs"],
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_steps=cfg["train"]["save_steps"],
        output_dir=out_dir,
        logging_dir=os.path.join(out_dir, "logs"),
        report_to=[],
        optim="adamw_8bit",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        formatting_func=fmt, max_seq_length=cfg["train"]["max_len"],
        packing=True, args=training_args,
    )

    class PushOnSave(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            last_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(last_ckpt):
                _push_folder(repos["a_ckpt_model"], last_ckpt)
            # Best effort merged snapshot for visibility
            try:
                merged = os.path.join(args.output_dir, f"merged_step_{state.global_step}")
                pathlib.Path(merged).mkdir(parents=True, exist_ok=True)
                FastModel.push_to_hub_merged(
                    model, tokenizer, save_directory=merged, repo_id=None,
                    token=os.environ["HF_TOKEN"], push_to_hub=False
                )
                _push_folder(repos["a_merged_model"], merged)
            except Exception as e:
                print(f"[warn] merge snapshot failed: {e}")
            return control

    trainer.add_callback(PushOnSave())
    resume_flag = os.path.isdir(out_dir) and any(p.name.startswith("checkpoint-") for p in pathlib.Path(out_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume_flag)

    # Final merged A
    final_dir = os.path.join(out_dir, "final_merged_A")
    pathlib.Path(final_dir).mkdir(parents=True, exist_ok=True)
    FastModel.push_to_hub_merged(
        model, tokenizer, save_directory=final_dir, repo_id=None,
        token=os.environ["HF_TOKEN"], push_to_hub=False
    )
    _push_folder(repos["a_merged_model"], final_dir)

    # Symlink for DIFF stage
    if os.path.islink("runs/A_final"):
        os.unlink("runs/A_final")
    os.makedirs("runs", exist_ok=True)
    os.system(f"ln -sfn {final_dir} runs/A_final")

if __name__ == "__main__":
    train()
