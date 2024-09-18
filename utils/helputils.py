import torch


def save_model(wandb, log_path, model_params_dict, save_wandb: bool = False) -> None:
    # model_path = f"runs/{run_name}/{args.exp_name}"
    print("### SAVING MODEL ###")
    for param in model_params_dict.keys():
        param_path = f"{log_path}/{param}.pt"

        # needs to be checked: it is needed to save the model locally before saving it to wandb
        torch.save(model_params_dict[param], param_path)
        print(f"{param} saved locally to: {param_path}")
        if save_wandb:
            wandb.save(param_path, policy="now")
            print(f"saving {param} to wandb")


def upload_csv_wandb(wandb, csv_file_path) -> None:
    print("### UPLOADING CSV TO WANDB ###")
    # wandb.log(csv_file_path) # wandb.log requires a dictionary
    wandb.save(csv_file_path, policy="now")
