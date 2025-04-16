#!/usr/bin/env python

import argparse
import os
import torch
import random
import json
import gc
import nltk
from nltk.corpus import wordnet as wn
from diffusers import FluxPipeline

# Ensure WordNet is loaded
nltk.download('wordnet', quiet=True)

# Generation parameters
guidance_scale = 0.0
max_sequence_length = 256

def load_generation_plan(json_path):
    with open(json_path, 'r') as f:
        raw_plan = json.load(f)

    generation_plan = {
        tuple(key.split("|")): count for key, count in raw_plan.items()
    }
    return generation_plan

def is_animal(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    for syn in synsets:
        for path in syn.hypernym_paths():
            if any("animal" in s.name().lower() for s in path):
                return True
    return False

def generate_prompt(subject1, subject2):
    return (
        f"A centered headshot of a {subject1} looking directly at the camera. "
        f"A second {subject2}, also a headshot, appears partially at the bottom, also centered and looking forward. "
        "Plain background. Only these two animal faces are visible."
    )

def main(output_dir, model_path, steps, plan_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Using device: {device}, dtype: {dtype}")

    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)

    n_gpus = torch.cuda.device_count()
    print(f"[INFO] Found {n_gpus} GPU(s)")

    if n_gpus > 1:
        pipe = torch.nn.DataParallel(pipe)
        pipe = pipe.cuda()
    else:
        pipe = pipe.to(device)

    os.makedirs(output_dir, exist_ok=True)
    generation_plan = load_generation_plan(plan_path)

    base_generator = torch.Generator(device=device)

    for (subject1, subject2), count in generation_plan.items():
        for i in range(0, count, batch_size):
            current_batch = min(batch_size, count - i)
            if current_batch == 0:
                continue  # safety check

            prompts = [generate_prompt(subject1, subject2) for _ in range(current_batch)]
            seeds = [random.randint(0, 2**64 - 1) for _ in range(current_batch)]
            generators = [base_generator.manual_seed(s) for s in seeds]

            print(f"Generating batch of {current_batch} images for: {subject1} & {subject2}")

            results = pipe(
                prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                max_sequence_length=max_sequence_length,
                generator=generators,
            )

            for img, seed in zip(results.images, seeds):
                filename = f"{subject1}_{subject2}_{seed}.png"
                path = os.path.join(output_dir, filename)
                img.save(path)
                print(f"Saved image to {path}")

            # Debug memory usage
            if device.type == "cuda":
                print(f"[DEBUG] CUDA Mem Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                print(f"[DEBUG] CUDA Mem Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

            # Cleanup to avoid memory buildup
            del results, prompts, generators, seeds
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate batched images with the FLUX.1-schnell model.")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--model-path", type=str, default="models/FLUX.1-dev", help="Path to the model directory")
    parser.add_argument("--steps", type=int, default=32, help="Number of inference steps")
    parser.add_argument("--plan", type=str, required=True, help="Path to the JSON generation plan file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")

    args = parser.parse_args()
    main(args.output_dir, args.model_path, args.steps, args.plan, args.batch_size)
