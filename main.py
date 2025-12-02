import soundata

#fazer pip install soundata
#
# -----------------------------------------------------
# 1. Inicializar o dataset UrbanSound8K
# -----------------------------------------------------
dataset = soundata.initialize('urbansound8k')

# -----------------------------------------------------
# 2. Fazer download (apenas uma vez)
#    soundata faz download para ~/.soundata automaticamente
# -----------------------------------------------------
print("A fazer download (se já existir, não volta a sacar)...")
dataset.download()
dataset.validate()

# -----------------------------------------------------
# 3. Escolher um clip aleatório
# -----------------------------------------------------
example_clip = dataset.choice_clip()

# Ver info básica
print(example_clip)

# -----------------------------------------------------
# 4. Aceder ao áudio
# -----------------------------------------------------
audio, sr = example_clip.audio
print("Forma do áudio:", audio.shape)
print("Sample rate:", sr)
# -----------------------------------------------------
# 5. Metadata útil
# -----------------------------------------------------
print("Classe ID:", example_clip.tags['source_file'])  # ficheiro original
print("Caminho do áudio:", example_clip.audio_path)
