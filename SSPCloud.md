# Setup SSP Cloud

## A chaque nouvelle session (le venv est détruit au redémarrage)

```bash
cd ~/work/Project-age-estimation-pytorch

# Recréer le venv dans /tmp (évite de remplir le disque de 9.8G)
uv venv --python 3.11 /tmp/venv
ln -s /tmp/venv venv
source venv/bin/activate

# Rediriger le cache uv vers /tmp (évite de remplir ~/work)
export UV_CACHE_DIR=/tmp/uv-cache

# Installer les dépendances
uv pip install -r requirements-linux.txt
uv pip install "numpy<2.0" "opencv-python-headless<4.9"

# Fix: setuptools>=71 ne fournit plus pkg_resources (requis par tensorboard)
uv pip install "setuptools<71"
```

## Lancer l'entraînement

```bash
python train.py --data_dir ./appa-real-release --tensorboard tf_log TRAIN.BATCH_SIZE 64
```

## TensorBoard

```bash
tensorboard --logdir=tf_log --bind_all
# Accessible via https://user-<username>.user.lab.sspcloud.fr/proxy/6006/
```

## Points importants

- **Batch size** : 64 max (128 = CUDA out of memory sur le GPU 14.5GB)
- **Disque** : 9.8G monté sur `~/work`. Vérifier régulièrement avec `df -h ~/work`
- **Cache** : `~/work/.cache` peut grossir vite, vider si nécessaire avec `rm -rf ~/work/.cache`
- **Corbeille** : `~/work/.Trash-1000` aussi, vider avec `rm -rf ~/work/.Trash-1000`
- **Checkpoints** : sauvegardés dans `checkpoint/`, chaque fichier fait ~295MB
