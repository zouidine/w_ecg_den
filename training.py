import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics.pairwise import cosine_similarity

def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N/D) * 100

    return PRD


def COS_SIM(y, y_pred):
    cos_sim = []
    y = np.squeeze(y, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim


def SNR(y1,y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)

    SNR = 10*np.log10(N/D)

    return SNR

def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out)-SNR(y_clean, y_in)

def train(model, config, train_loader, device, valid_loader=None,
          valid_epoch_interval=5, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])

    if foldername != "":
        best_path = foldername + "/bestmodel.pth"

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=.1, verbose=True
    )

    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch = clean_batch.to(device)
                noisy_batch = noisy_batch.to(device)
                optimizer.zero_grad()

                loss = model(clean_batch, noisy_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
        lr_scheduler.step()
        if valid_loader is not None and (epoch_no+1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch,
                                   noisy_batch) in enumerate(it, start=1):
                        clean_batch = clean_batch.to(device)
                        noisy_batch = noisy_batch.to(device)
                        loss = model(clean_batch, noisy_batch)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid/batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )

            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\nbest loss is updated to ",avg_loss_valid / batch_no,
                      "at", epoch_no,"\n")

                if foldername != "":
                    torch.save(model.state_dict(), best_path)

def evaluate(model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0

    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch = clean_batch.to(device)
            noisy_batch = noisy_batch.to(device)

            if shots > 1:
                output = 0
                for i in range(shots):
                    output += model.denoising(noisy_batch)
                output /= shots
            else:
                output = model.denoising(noisy_batch) #B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()


            eval_points += len(output)
            ssd_total += np.sum(SSD(clean_numpy, out_numpy))
            mad_total += np.sum(MAD(clean_numpy, out_numpy))
            prd_total += np.sum(PRD(clean_numpy, out_numpy))
            cos_sim_total += np.sum(COS_SIM(clean_numpy, out_numpy))
            snr_noise += np.sum(SNR(clean_numpy, noisy_numpy))
            snr_recon += np.sum(SNR(clean_numpy, out_numpy))
            snr_improvement += np.sum(SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
            restored_sig.append(out_numpy)

            it.set_postfix(
                ordered_dict={
                    "ssd_total": ssd_total/eval_points,
                    "mad_total": mad_total/eval_points,
                    "prd_total": prd_total/eval_points,
                    "cos_sim_total": cos_sim_total/eval_points,
                    "snr_in": snr_noise/eval_points,
                    "snr_out": snr_recon/eval_points,
                    "snr_improve": snr_improvement/eval_points,
                },
                refresh=True,
            )

    restored_sig = np.concatenate(restored_sig)

    print("\nssd_total: ",ssd_total/eval_points)
    print("mad_total: ", mad_total/eval_points,)
    print("prd_total: ", prd_total/eval_points,)
    print("cos_sim_total: ", cos_sim_total/eval_points,)
    print("snr_in: ", snr_noise/eval_points,)
    print("snr_out: ", snr_recon/eval_points,)
    print("snr_improve: ", snr_improvement/eval_points,)
