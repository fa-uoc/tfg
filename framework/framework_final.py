import os
import torch
import torch.nn as nn
import torchaudio.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchaudio import sox_effects
# ResNet1D descargado de git https://github.com/hsd1503/resnet1d
from infrastructure.resnet1d import resnet1d
import torchaudio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
#from tqdm import tqdm
from tqdm.notebook import tqdm
import time
import pandas as pd
from collections import defaultdict
import random
import gc

#from infrastructure.signal_grad_cam.signal_grad_cam.pytorch_cam_builder import TorchCamBuilder
from signal_grad_cam.pytorch_cam_builder import TorchCamBuilder
from PIL import Image

# Necesario activar en entorno tfg2, que no da error de GPU al importar tensorflow (driver 580 con la GTX1060).
# No se aprecia mejora significativa de rendimiento, por lo que nos quedamos con el entorno tfg original
# Descomentar si necesario
#torch.backends.cudnn.enabled = False

#Variable de entorno requerida por tensorflow para reproducibilidad
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Valor para margen de seguridad en operaciones en denominador
EPSILON = 1e-9

# DEFAULTS

# Rutas locales
FINAL_PATH = './final'
CHECKPOINTS_PATH = './checkpoints'
XAI_PATH = '/mnt/data/GSC/XAI'
BACKGROUND_NOISES = sorted(Path(f"/mnt/data/GSC/speech_commands_v0.02/_background_noise_/").glob('*wav'))
SAVE_EVERY = 2

# PyTorch
DEVICE = torch.device('cuda')
EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
NUM_CLASSES = 35

# Audio
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = int(400)
WIN_LENGTH = N_FFT
HOP_LENGTH = int(WIN_LENGTH/2)

class Persister:
    """
    Clase encargada de la escritura en disco
    """
    def __init__(self,
                 final_path: str = FINAL_PATH,
                 checkpoints_path: str = CHECKPOINTS_PATH,
                 xai_path: str = XAI_PATH):
        self.final_path = Path(final_path)
        self.final_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = Path(checkpoints_path)
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.xai_path = Path(xai_path)
        self.xai_path.mkdir(parents=True, exist_ok=True)
        

    def new_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if checkpoint['is_best']:
            path = self.checkpoints_path / f"best_checkpoint_epoch_{checkpoint['epoch']:02d}.pt"
        else: 
            path = self.checkpoints_path / f"checkpoint_epoch_{checkpoint['epoch']:02d}.pt"
        torch.save(checkpoint, path)
    def final_model(self, final_model: dict[str, object]) -> None:
        path = self.final_path / f"final_model.pt"
        torch.save(final_model, path)
    def save_spectrogram_images(self,
                                specs: torch.Tensor,
                                file_names: list[str],
                                dpi: int = 150,
                                cmap: str = "viridis"):
        if specs.dim() == 4:
            specs = specs.squeeze(1)
        specs = specs.detach().cpu().numpy()
        assert specs.shape[0] == len(file_names)

        for i, fname in enumerate(file_names):
            out_path = self.xai_path / "espectrogramas" / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            spec = specs[i]

            fig, ax = plt.subplots()
            ax.imshow(spec, origin='lower', aspect='auto', cmap=cmap)
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig)
    def save_resampled_waves(self,
                                waves: torch.Tensor,
                                file_names: list[str]):

        for i, fname in enumerate(file_names):
            out_path = self.xai_path / "espectrogramas" / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            spec = specs[i]

            fig, ax = plt.subplots()
            ax.imshow(spec, origin='lower', aspect='auto', cmap=cmap)
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig)

            
class MetricsCalculator:
    """
    Clase para el cálculo de métricas durante el entrenamiento.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()
    def reset(self):
        self.y_true = []
        self.y_predictions = []
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        #Tomamos de las clases (en dim=1) la de mayor valor para computar en cpu
        preds = predictions.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        self.y_predictions.extend(preds.tolist())
        self.y_true.extend(labels.tolist())
    def get_accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_predictions)) if self.y_true else 0.0
    def get_prfs(self, average: str = None) -> np.ndarray:
        return precision_recall_fscore_support(self.y_true, self.y_predictions, labels=list(range(self.num_classes)), average=average, zero_division=0)
    def get_cm(self):
        return confusion_matrix(self.y_true, self.y_predictions, labels=list(range(self.num_classes)))

class AudioResNetModel:
    """
    Clase para obtener un modelo ResNet adaptado a audio
    dims: 2 -> resnet18 adaptado para espectrogramas
    dims: 1 -> ResNet1D para trabajo con audio en crudo
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dims: int = 2):
        assert dims in (1,2), "dims debe ser un entero de valor 1 o 2"
        self.dims = dims
        self.num_classes = NUM_CLASSES
        
    def get_model(self):
        # Espectrogramas: ResNet18 adaptado
        if self.dims == 2:
            model_2d = models.resnet18(weights=None)
            model_2d.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model_2d.fc = nn.Linear(model_2d.fc.in_features, NUM_CLASSES)
            return model_2d
        # Audio en crudo: ResNet1D
        if self.dims == 1:
            # Modelo
            model_1d = resnet1d.ResNet1D(in_channels=1,
                  base_filters=32, # reducimos a 32 filtros con respecto al modelo 2D por la gran diferencia en cómputo
                  kernel_size=7,
                  stride=2,           
                  groups=1,           
                  n_block=8,          
                  n_classes=NUM_CLASSES)
            return model_1d
        

# Dataset
class GSC(Dataset):
    """
    - file_list: lista de paths para acceso al particionado (requiere rutas absolutas)
    - labels: etiquetas
    """
    def __init__(self,
                 file_list: List[str],
                 labels: List[int]):
        self.files = list(file_list)
        self.labels = list(labels)
    def __len__(self):
        return len(self.files)
    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sample_rate = torchaudio.load(path)
        return wav
    def __getitem__(self, index) -> Tuple[torch.Tensor, int, str]:
        path = self.files[index]
        wav = self._load_audio(path)
        label = self.labels[index]
        return wav, label, str(path)

#collate_fn (llamada desde el DataLoader)
def prepare_batch(batch, max_len: int = int(1.0 * SAMPLE_RATE)):
    """
    - batch: lista (wav, label, path)
    """
    max_len = int(max_len)
    waves, labels, paths = zip(*batch)
    lengths = [w.shape[-1] for w in waves]
    padded = []
    for w in waves:
        t = w.shape[-1]
        if t >= max_len:
            padded_w = w[:, :max_len]
        else:
            pad_len = max_len - t
            padded_w = torch.nn.functional.pad(w, (0, pad_len))
        padded.append(padded_w)
    batch_waves = torch.stack(padded, dim=0)
    return batch_waves, torch.tensor(labels, dtype=torch.long), list(paths)

#Preprocesa audio
class Preprocessor:
    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 n_mels: int = N_MELS,
                 win_length: int = WIN_LENGTH,
                 hop_length: int = HOP_LENGTH,
                 n_fft: int = N_FFT,
                 with_log: bool = True,
                 device: str = str(DEVICE),
                 # Por defecto modo test (sin aumentaciones) para no romper el código anterior cuando integremos
                 mode: str = "test",
                 # Probabilidades de una aumentación. Por defecto desactivadas.
                 augmentation_probs: Optional[dict[str, float]] = None,
                 # Opciones de salida: "mel" (por defecto) o "waveform"
                 output: str = "mel"): 

        assert output in ('mel', 'waveform'), "output admite como valores 'mel' o 'waveform'"

        self.mode = mode
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.with_log = with_log
        self.device = device
        self.augmentation_probs = augmentation_probs or {}
        self.background_noises = BACKGROUND_NOISES
        self.output = output

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            n_mels = self.n_mels,
            power = 2.0)
        self.mel.to(self.device)
        self.db = torchaudio.transforms.AmplitudeToDB(stype='power').to(self.device) if with_log else None
    
    # Aumentaciones

    def _additive_noise( self, waveform: torch.Tensor,
                         snr_db_min: float = 18.0,
                         snr_db_max: float = 30.0):

        snr_db = random.uniform(snr_db_min, snr_db_max) # Elegimos SNR de manera aleatoria entre min y max

        noise = torch.randn_like(waveform) # Ruido blanco

        augmented = F.add_noise(waveform, noise,
                                torch.tensor([snr_db], device=waveform.device))

        return augmented.clamp(-1.0, 1.0)    

    def _random_gain(self, waveform, gain_db_min=-6.0, gain_db_max=6.0):

        gain_db = random.uniform(gain_db_min, gain_db_max)

        augmented = F.gain(waveform, gain_db=gain_db)

        return augmented.clamp(-1.0, 1.0)
    

    def _basic_eq_and_compress(self, waveform: torch.Tensor):
        wave = waveform.float().to('cpu')
        effects = [["highpass", "180"],
                   ["equalizer", "250", "0.3", "-3"],
                   ["compand", "20.0,120.0", "6:-20.0,-15.0"]]

        output, _ = sox_effects.apply_effects_tensor(wave, self.sample_rate, effects)
        return output.to(self.device)
    
    def _time_shift(self, waveform: torch.Tensor, shift_limit: float = 0.05): 
        length = waveform.size(-1) # Esperamos el número de samples en la última dimensión del tensor
        shift = int(random.uniform(-shift_limit, shift_limit) * length) # Escalamos la longitud por un número aleatorio en el umbral que define shift
        return torch.roll(waveform, shifts=shift, dims=-1) # Implementación naive sólo para aumentación: Rota según el signo de shift hacia derecha o izquierda el contenido para no introducir silencio (pero da lugar a repetición del principio o el final del archivo)

    def _time_stretch(self, waveform: torch.Tensor):
        wave = waveform.float().to('cpu')
        rate = random.uniform(0.9, 1.1)
        effects = [["tempo", str(rate)],
                   ["pad", "0", "1"],
                   ["trim", "0", "1"]]

        output, _ = sox_effects.apply_effects_tensor(wave, self.sample_rate, effects)
        return output.to(self.device)

    def _speed_perturb(self, waveform: torch.Tensor):
        wave = waveform.float().to('cpu')
        speed_chg = random.uniform(0.9, 1.1)
        effects = [["speed", str(speed_chg)],
                   ["rate", str(self.sample_rate)],
                   ["pad", "0", "1"],
                   ["trim", "0", "1"]]

        output, _ = sox_effects.apply_effects_tensor(wave, self.sample_rate, effects)
        return output.to(self.device)
        
    def _mix_background(self, waveform: torch.Tensor, background: Path, snr_db: float):
        noise, _ = torchaudio.load(str(background))
        noise = noise.float().to(self.device)
        T = waveform.shape[-1]
        # Los backgrounds del dataset son siempre mayores: recortamos a la longitud de w y no nos preocupamos de otros casos
        if noise.shape[-1] > T:
            noise = noise[:,:T]
        snr = torch.tensor([snr_db], device=waveform.device)

        mixed = F.add_noise(waveform, noise, snr)

        return mixed

    def _reverb(self, waveform: torch.Tensor):
        wave = waveform.float().to('cpu')        
        rev_min = 40.0
        rev_max = 60.0
        rev_level = random.uniform(rev_min, rev_max)
        effects = [["reverb", str(rev_level)],
                   ["remix", "1-2"]]

        output, _ = sox_effects.apply_effects_tensor(wave, self.sample_rate, effects)
        return output.to(self.device)
    
    def _frequency_masking(self, mels: torch.Tensor, freq_mask_param: int = 6):
        freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param)
        return freq_mask(mels)

    def _time_masking(self, mels: torch.Tensor, time_mask_param: int = 10):
        time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param)
        return time_mask(mels)
    
    def transform_batch(self, batch_waves: torch.Tensor) -> torch.Tensor:
        x = batch_waves.float().to(self.device)

        # Aumentaciones en waveform
        if self.mode == "training":
            augmented_batch = []
            for wave in x:

                p = self.augmentation_probs.get('additive_noise', 0.0)
                if random.random() < p:
                    wave = self._additive_noise(wave)

                p = self.augmentation_probs.get('random_gain', 0.0)
                if random.random() < p:
                    wave = self._random_gain(wave)

                p = self.augmentation_probs.get('basic_eq_and_compress', 0.0)
                if random.random() < p:
                    wave = self._basic_eq_and_compress(wave)
                
                p = self.augmentation_probs.get('time_shift', 0.0)
                if random.random() < p:
                    wave = self._time_shift(wave)

                p = self.augmentation_probs.get('time_stretch', 0.0)
                if random.random() < p:
                    wave = self._time_stretch(wave)
                
                p = self.augmentation_probs.get('speed_perturb', 0.0)
                if random.random() < p:
                    wave = self._speed_perturb(wave)

                p = self.augmentation_probs.get('mix_background', 0.0)
                if random.random() < p:
                    wave = self._mix_background(wave, random.choice(self.background_noises), 40)
                    
                p = self.augmentation_probs.get('reverb', 0.0)
                if random.random() < p:
                    wave = self._reverb(wave)
                
                augmented_batch.append(wave)
            x = torch.stack(augmented_batch)

        # Si output: waveform, retornamos waveform
        if self.output == "waveform":
            return x

        # Si output: mel
        mels = self.mel(x)
        if mels.dim() == 4:
            mels = mels.squeeze(1)
        # log-mel
        if self.db:
            mels = self.db(mels)

        # Normalización sólo por frecuencia
        mean = mels.mean(dim=2, keepdim=True)
        std = mels.std(dim=2, keepdim=True)
        mels = (mels - mean) / (std + EPSILON)

        # Aumentaciones sobre espectrogramas
        augmented_mels=[]
        if self.mode == "training":
            for mel in mels:
                p = self.augmentation_probs.get('frequency_masking', 0.0)
                if random.random() < p:
                    mel = self._frequency_masking(mel)

                p = self.augmentation_probs.get('time_masking', 0.0)
                if random.random() < p:
                    mel = self._time_masking(mel)

                augmented_mels.append(mel)

            mels = torch.stack(augmented_mels)

        mels = mels.unsqueeze(1) #forma x: [BATCH, CHANNEL, TIME]

        return mels        
      
    def save_spectrogram_images(self,
                                batch_waves: torch.Tensor,
                                file_paths: list,
                                dpi: int = 150,
                                cmap: str = "viridis",
                                mels_in_db: bool = True):


        assert len(file_paths) == batch_waves.size(0)
        
        x = batch_waves.float().to(self.device)
        mels = self.mel(x)
        if mels.dim() == 4:
            mels = mels.squeeze(1)
        if not mels_in_db and self.db:
            mels_db = self.db(mels)
        else:
            mels_db = mels
        mels_db = mels_db.detach().cpu()

        for i in range(mels_db.size(0)):
            spec = mels_db[i].numpy()
            fig, ax = plt.subplots(figsize=(spec.shape[1]/100, spec.shape[0]/100), dpi=dpi)
            im = ax.imshow(spec, origin='lower', aspect='auto', cmap=cmap)
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            out_path = file_paths[i]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

#Trainer con cómputo de loss por batches
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 preprocessor: Preprocessor,
                 persister: Persister,
                 device: str = DEVICE,
                 learning_rate: float = LEARNING_RATE,
                 epochs: int = EPOCHS,
                 num_classes: int = NUM_CLASSES,
                 save_every = SAVE_EVERY,
                 resume_from: Optional[str] = None):
    
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.preprocessor = preprocessor
        self.persister = persister
        #Optimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        #Específicos
        self.epochs = epochs
        self.num_classes = num_classes
        #Métricas
        self.history = {'train_loss': [],
                        'train_acc': [],
                        'train_f1': [],
                        'val_loss': [],
                        'val_acc': [],
                        'val_f1': [],
                        'epoch_time': []
                        }
        self.save_every = int(save_every)
        self.best_val_f1 = -1
        #Para continuar entrenamiento desde cierto punto
        self.resume_from = resume_from
        if self.resume_from:
            self._load_checkpoint(self.resume_from)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'is_best': is_best
        }
        self.persister.new_checkpoint(checkpoint)

    def _load_checkpoint(self, path: str):
        device = self.device
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_f1 = checkpoint.get('best_val_f1', self.best_val_f1)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Cargado desde {path}. Comenzamos en epoch {start_epoch}")
        return start_epoch

    def _save_final_model(self, epoch:int):
        final_model = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }
        self.persister.final_model(final_model)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        accumulated_batch_samples = 0
        metrics = MetricsCalculator(self.num_classes)

        for waves, labels, paths in tqdm(self.train_loader, desc="Batches entrenamiento", leave = False):
            audios = self.preprocessor.transform_batch(waves)
            audios.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(audios)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #Calculamos con labels.size(0) en lugar de BATCH_SIZE por si el último batch es más pequeño
            accumulated_batch_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            metrics.update(preds, labels)
        avg_loss = total_loss / accumulated_batch_samples if accumulated_batch_samples > 0 else 0
        accuracy = metrics.get_accuracy()
        _, _ , avg_f1, _ = metrics.get_prfs(average='macro')
        return avg_loss, accuracy, avg_f1
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        accumulated_batch_samples = 0
        metrics = MetricsCalculator(self.num_classes)

        #desactivamos autograd para validación durante entrenamiento
        with torch.no_grad():
            for waves, labels, paths in tqdm(self.val_loader, desc="Batches validación", leave=False):
                mels = self.preprocessor.transform_batch(waves)
                labels = labels.to(self.device)
                preds = self.model(mels)
                loss = self.criterion(preds, labels)
                accumulated_batch_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)
                metrics.update(preds, labels)
        avg_loss = total_loss/accumulated_batch_samples if accumulated_batch_samples > 0 else 0
        accuracy = metrics.get_accuracy()
        _, _ , avg_f1, _ = metrics.get_prfs('macro')
        return avg_loss, accuracy, avg_f1
        
    def fit(self,
            resume: bool = False,
            start_epoch: int = 1):
        if resume and self.resume_from:
            start_epoch = self._load_checkpoint(self.resume_from)
        for epoch in tqdm(range(start_epoch, self.epochs + 1), desc="Epoch:"):
            start = time.time()
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            epoch_time = time.time() - start
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)            
            self.history['epoch_time'].append(epoch_time)
            
            print(f"Epoch {epoch}/{self.epochs}")
            print(f"  Train - Pérdida: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1-score: {train_f1:.4f}")
            print(f"  Val   - Pérdida: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")
    
            if self.save_every > 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self._save_checkpoint(epoch, is_best=True)
        self._save_final_model(epoch)

class Evaluator:
    def __init__(self,
                 model: nn.Module,
                 test_loader: DataLoader,
                 preprocessor: Preprocessor,
                 device: str = DEVICE,
                 num_classes: int = NUM_CLASSES):
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)        
        self.test_loader = test_loader
        self.preprocessor = preprocessor
        self.num_classes = NUM_CLASSES
        self.criterion = nn.CrossEntropyLoss()
        
    def eval(self): #finalmente se encarga signal_grad_cam. Se puede probablemene aligerar con no_grad
        self.model.eval()
        total_loss = 0.0
        accumulated_batch_samples = 0
        metrics = MetricsCalculator(self.num_classes)

        with torch.no_grad():
            for waves, labels, paths in tqdm(self.test_loader, desc="Batches evaluación", leave=False):
                mels = self.preprocessor.transform_batch(waves)
                mels.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(mels)
                loss = self.criterion(preds, labels)
                accumulated_batch_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)
                metrics.update(preds, labels)
        avg_loss = total_loss/accumulated_batch_samples if accumulated_batch_samples > 0 else 0
        accuracy = metrics.get_accuracy()
        precission, recall, f1, support = metrics.get_prfs('macro')
        cm = metrics.get_cm()
        return cm, precission, recall, avg_loss, accuracy, f1, support

    def eval_waveform_shape(self):
        self.model.eval()
        total_loss = 0.0
        accumulated_batch_samples = 0
        metrics = MetricsCalculator(self.num_classes)
        
        with torch.no_grad():
            for waves, labels, paths in tqdm(self.test_loader, desc="Batches evaluación", leave=False):
                waves = self.preprocessor.transform_batch(waves)
                waves = waves.to(self.device).float()
                if waves.dim() == 2:
                    waves = waves.unsqueeze(1)
                waves.requires_grad_(True)
                labels = labels.to(self.device)
                preds = self.model(waves)
                loss = self.criterion(preds, labels)
                accumulated_batch_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)
                metrics.update(preds, labels)
        avg_loss = total_loss/accumulated_batch_samples if accumulated_batch_samples > 0 else 0
        accuracy = metrics.get_accuracy()
        precission, recall, f1, support = metrics.get_prfs('macro')
        cm = metrics.get_cm()
        return cm, precission, recall, avg_loss, accuracy, f1, support
        
    def eval_samples(self):
        self.model.eval()
        torch.set_grad_enabled(True)
        results = []
        idx_base = 0
        with torch.no_grad():
            for batch_idx, (batch_waves, batch_labels, batch_paths) in enumerate(self.test_loader):
                mels = self.preprocessor.transform_batch(batch_waves)
                mels = mels.to(self.device)
                batch_labels = batch_labels.to('cpu').numpy()
                # Forward
                logits = self.model(mels)                         # (B, num_clases)
                #probabilidades finales
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C)
                #predicciones realizadas
                preds = probs.argmax(axis=1)
                #mayor probabilidad
                top1 = probs.max(axis=1)
                #segunda mayor probabilidad
                top2 = np.array([np.partition(p, -2)[-2] if p.size>1 else 0.0 for p in probs])
                batch_length = len(batch_waves)
                for i in range(batch_length):
                    results.append({
                        'idx': idx_base + i,
                        'path': batch_paths[i],
                        'real': int(batch_labels[i]),
                        'predicción': int(preds[i]),
                        'top1_prob': float(top1[i]),
                        'top2_prob': float(top2[i]),
                        'margen': top1[i] - top2[i],
                        'probs': probs[i].tolist()
                    })
                idx_base += batch_length
        dataframe = pd.DataFrame(results)
        return dataframe

# Funciones auxiliares

# Grid de imágenes
def show_image_grid(paths, rows=2, cols=3, figsize=(12,8)):
    paths = [Path(p) for p in paths]
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i in range(n):
        ax = axes[i]
        ax.axis('off')
        if i < len(paths):
            img = Image.open(paths[i])
            ax.imshow(img)
    plt.tight_layout()
    plt.show()

# Cálculo y persistencia de CAMs con signal-grad-cam (wrapper)
def signal_grad_cam_overlays2d(model: nn.Module, dataloader: DataLoader,
                              preprocessor: Preprocessor, xai_path: str, explainer: str, label_names: np.ndarray,
                              target_layers: List[str], target_classes: List[int], targets_list: List[int], is_1D_signal: bool = False):
    '''Motivos del wrapper:
    - Para que sigal-grad-cam renderice muestra a muestra los overlays hay que pasar los archivos y los targets uno a uno, si no los apila en el mismo gráfico.
    - Falla la ejecución de la librería en ejecuciones sucesivas en el sistema cuando se pasan varios target layers como lista. Se sospecha de problemas de acumulación de gradientes en alguna función, pero no se ha identificado. Se rodea el problema instanciando cam_builder para cada target_layer.
    Se configura por defecto get_cams() con softmax_final = False'''
    idx = 0
    for batch_waves, batch_labels, batch_paths in dataloader:
        # Esta variable hace falta para acceder a la clave del diccionario de cams para los overlays
        current_targets = targets_list[idx:idx+batch_waves.shape[0]]
        # Espectrogramas de entrada
        mels = preprocessor.transform_batch(batch_waves)
        # la implementación de signal_grad_cam espera un numpy
        mels_cpu = mels.detach().cpu().numpy().astype(np.float32)

        data_list = list(mels_cpu)
        data_labels = batch_labels.tolist()

        # Cálculo y persistencia de CAMs
        # Instanciamos builder por target_layer para
        for target_layer in target_layers:
            cam_builder = TorchCamBuilder(model = model, transform_fn = None, use_gpu = True, class_names = label_names, time_axs = 2, input_transposed = False, ignore_channel_dim = False, is_regression_network = False)
            
            cams, predicted_probs, bar_ranges = cam_builder.get_cam(
                data_list = data_list,
                data_labels = data_labels,
                target_classes = target_classes,
                results_dir_path = f"{xai_path}/CAMs/{idx}-{idx+batch_waves.shape[0]-1}",
                explainer_types = explainer,
                target_layers = target_layer,
                softmax_final = False)

            for i in range(len(data_list)):
                # Carga de CAMs guardadas para cálculo de overlays
                single_cams = {f"{explainer}_{target_layer}_class{current_targets[i]}": [cams[f"{explainer}_{target_layer}_class{current_targets[i]}"][i]]}
                single_probs = {f"{explainer}_{target_layer}_class{current_targets[i]}": [predicted_probs[f"{explainer}_{target_layer}_class{current_targets[i]}"][i]]}

                path_str = f"{xai_path}/CAMs/overlays/{idx}-{idx+batch_waves.shape[0]-1}/item{i}"
                Path(path_str).mkdir(parents=True, exist_ok=True)

                # Cálculo y persistencia de overlays
                cam_builder.overlapped_output_display(
                    data_list = [data_list[i]],
                    data_labels = [data_labels[i]],
                    target_layers = target_layer,
                    predicted_probs_dict = single_probs,
                    bar_ranges_dict = bar_ranges,
                    cams_dict = single_cams,
                    explainer_types = explainer,
                    target_classes = [current_targets[i]],
                    results_dir_path = path_str,
                    data_sampling_freq = SAMPLE_RATE,
                    dt = 0.01)
                
        del cam_builder
        cam_builder = None
        torch.cuda.empty_cache()
        gc.collect()
        idx += batch_waves.shape[0]

def signal_grad_cam_single1d(model: nn.Module, dataloader: DataLoader,
                              preprocessor: Preprocessor, xai_path: str, explainer: str, label_names: np.ndarray,
                              target_layers: List[str], target_classes: List[int], targets_list: List[int]):
    idx = 0
    for batch_waves, batch_labels, batch_paths in dataloader:
        current_targets = targets_list[idx:idx+batch_waves.shape[0]]
        waves = preprocessor.transform_batch(batch_waves)
        waves_cpu = waves.detach().cpu().numpy().astype(np.float32)

        data_list = list(waves_cpu)
        data_labels = batch_labels.tolist()
    for target_layer in target_layers:
        cam_builder = TorchCamBuilder(model=model,
                                      transform_fn = None,
                                      use_gpu = True,
                                      class_names = label_names,
                                      time_axs=1,
                                      input_transposed=False,
                                      ignore_channel_dim = False,
                                      is_regression_network= False)
                    
        cams, predicted_probs, bar_ranges = cam_builder.get_cam(
            data_list = data_list,
            data_labels = data_labels,
            target_classes = target_classes,
            results_dir_path = f"{xai_path}/CAMs/",
            explainer_types = explainer,
            target_layers = target_layer,
            softmax_final = False)

        # Carga CAMs para visualización
        for i in range(len(data_list)):
                single_cams = {f"{explainer}_{target_layer}_class{current_targets[i]}": [cams[f"{explainer}_{target_layer}_class{current_targets[i]}"][i]]}
                single_probs = {f"{explainer}_{target_layer}_class{current_targets[i]}": [predicted_probs[f"{explainer}_{target_layer}_class{current_targets[i]}"][i]]}

                #single_channel
                path_str = f"{xai_path}/CAMs/single_channel/"
                Path(path_str).mkdir(parents=True, exist_ok=True)

                cam_builder.single_channel_output_display(
                    data_list=[data_list[i]],
                    data_labels = [data_labels[i]],
                    predicted_probs_dict = single_probs,
                    cams_dict = single_cams,
                    explainer_types=explainer,
                    target_classes = [current_targets[i]],
                    target_layers = target_layer,
                    desired_channels = None,
                    bar_ranges_dict = bar_ranges,
                    results_dir_path = path_str,
                    data_sampling_freq = SAMPLE_RATE,
                    marker_width = 15,
                    axes_names=("Time", "Amplitude"))

        del cam_builder
        cam_builder = None
        torch.cuda.empty_cache()
        gc.collect()
        idx += batch_waves.shape[0]


def signal_grad_cam_single2d(model: nn.Module, dataloader: DataLoader,
                              preprocessor: Preprocessor, xai_path: str, explainer: str, label_names: np.ndarray,
                              target_layers: List[str], target_classes: List[int], targets_list: List[int]):
    idx = 0
    for batch_waves, batch_labels, batch_paths in dataloader:
        current_targets = targets_list[idx:idx+batch_waves.shape[0]]
        waves = preprocessor.transform_batch(batch_waves)
        waves_cpu = waves.detach().cpu().numpy().astype(np.float32)

        data_list = list(waves_cpu)
        data_labels = batch_labels.tolist()
    for target_layer in target_layers:
        cam_builder = TorchCamBuilder(model=model,
                                      transform_fn = None,
                                      use_gpu = True,
                                      class_names = label_names,
                                      time_axs=2,
                                      input_transposed=False,
                                      ignore_channel_dim = False,
                                      is_regression_network= False)
                    
        cams, predicted_probs, bar_ranges = cam_builder.get_cam(
            data_list = data_list,
            data_labels = data_labels,
            target_classes = target_classes,
            results_dir_path = f"{xai_path}/CAMs/",
            explainer_types = explainer,
            target_layers = target_layer,
            softmax_final = False)

        # Carga CAMs para visualización
        for i in range(len(data_list)):
                single_cams = {f"{explainer}_{target_layer}_class{current_targets[i]}": [cams[f"{explainer}_{target_layer}_class{current_targets[i]}"][i]]}
                single_probs = {f"{explainer}_{target_layer}_class{current_targets[i]}": [predicted_probs[f"{explainer}_{target_layer}_class{current_targets[i]}"][i]]}

                #single_channel
                path_str = f"{xai_path}/CAMs/single_channel/"
                Path(path_str).mkdir(parents=True, exist_ok=True)

                cam_builder.single_channel_output_display(
                    data_list=[data_list[i]],
                    data_labels = [data_labels[i]],
                    predicted_probs_dict = single_probs,
                    cams_dict = single_cams,
                    explainer_types=explainer,
                    target_classes = [current_targets[i]],
                    target_layers = target_layer,
                    desired_channels = None,
                    bar_ranges_dict = bar_ranges,
                    results_dir_path = path_str,
                    data_sampling_freq = SAMPLE_RATE,
                    marker_width = 15,
                    axes_names=("Time", "Frequency"))

        del cam_builder
        cam_builder = None
        torch.cuda.empty_cache()
        gc.collect()
        idx += batch_waves.shape[0]
        
    
def plot_curves(model_dict: Dict[str, Any], title: str = 'Curvas'):
    history = model_dict['history']
    epochs = np.arange(1, model_dict['epoch']+1)
    plt.figure(figsize=(12,8))
    plt.title(title)
    
    ## Pérdida
    plt.subplot(3,1,1)
    plt.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='C0')
    plt.plot(epochs, history['val_loss'], 's--',  label='Val Loss', color='C0')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    ## Accuracy
    plt.subplot(3,1,2)
    plt.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='C1')
    plt.plot(epochs, history['val_acc'], 's--',  label='Val Acc', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    ## F1-score
    plt.subplot(3,1,3)
    plt.plot(epochs, history['train_f1'], 'o-', label='Train F1', color='C2')
    plt.plot(epochs, history['val_f1'], 's--',  label='Val F1', color='C2')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
