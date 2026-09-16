# rPPG — Residência Pipos

Sistema de fotopletismografia remota (rPPG): a partir de um vídeo do rosto
(captura via webcam ou arquivo já gravado), estima a **frequência cardíaca**, a
**variabilidade da frequência cardíaca (HRV)** e métricas de qualidade do
sinal. 

---

## Como funciona

1. **Detecção facial** — MediaPipe Face Landmarker localiza os pontos do rosto em cada frame.
2. **Extração das ROIs** — testa, bochecha esquerda, bochecha direita e glabela são recortadas; a média espacial de R, G, B é calculada dentro de cada uma.
3. **Pré-processamento** — suavização para remover ruído de câmera/movimento.
4. **Extração do sinal rPPG** — quatro algoritmos rodam em paralelo sobre cada ROI:
   - **CHROM** (De Haan & Jeanne, 2013)
   - **POS** (Wang et al., 2017) 
   - **GREEN** (Verkruysse et al., 2008)
   - **ICA** (Poh, McDuff & Picard, 2010)
5. **Combinação** — os sinais são padronizados (z-score), têm a polaridade alinhada e são combinados por pesos configuráveis: primeiro entre algoritmos dentro de cada ROI, depois entre ROIs. A combinação é feita em sinais, nunca em médias de HR.
6. **Filtragem final** — um único filtro Butterworth passa-banda restringe o sinal combinado à faixa fisiológica de frequência cardíaca (42–240 bpm). Esse mesmo sinal filtrado alimenta HR, HRV e as métricas finais.
7. **Biomarcadores** — a frequência cardíaca (pico espectral via FFT), HRV (SDNN, RMSSD, pNN50 a partir dos intervalos entre picos) e métricas de qualidade do sinal (SNR, concentração espectral, amplitude, energia) são calculados.
8. **Relatório** — resultado final, benchmarks e auditoria concisa de proveniência são impressos no console. Os benchmarks são observacionais e não escolhem o sinal final.

---

## Pré-requisitos

- Python 3.8+
- pip
- webcam (para captura) 

---

## Instalação

```bash
git clone https://github.com/AdaoRios/rPPG---Estatisca.git
cd rPPG
pip install -r requirements.txt
```

O modelo de landmarks faciais do MediaPipe já está incluído em
`assets/face_landmarker.task`.

---

## Como usar

Execute a partir desta pasta (`rPPG/`):

```bash
# Menu interativo (captura, análise, ou ambos)
python main.py

# Analisar um vídeo já existente
python main.py --video "data/captures/video_2026-01-01_10-00-00.mp4"

# Só capturar (sem analisar)
python main.py --capture-only

# Capturar e analisar em sequência, com duração e câmera customizadas
python main.py --duration 20 --camera 1
```

### Captura guiada por qualidade

Ao iniciar uma captura (`python main.py --capture-only`), a webcam abre em
**PREVIEW**. Esse estágio não grava vídeo e não exige estabilidade: use-o para
posicionar-se, ajustar a distância e movimentar-se livremente. Pressione
**ENTER** para solicitar o início ou **Q** para sair.

Ao pressionar ENTER, o sistema cria o MP4 e entra imediatamente em
**CAPTURING**: não existe quality gate antes da gravação. Movimento e
enquadramento facial são monitorados em tempo real somente para feedback.

Somente PREVIEW fica fora do arquivo. A duração solicitada começa ao pressionar
ENTER. Durante CAPTURING, todos os frames recebidos da câmera são gravados,
inclusive quando há movimento, face fora do enquadramento ou face não detectada;
essas situações geram aviso visual, nunca descarte, pausa ou reinício da
gravação. Ao fim da duração, ou com **Q**, o MP4 é finalizado e pode seguir para
`analyze_video.py` normalmente.

O overlay mantém valores de depuração de movimento e de enquadramento (tamanho,
centro e bounding box) para a calibração experimental.

Durante o **PREVIEW**, o overlay já mostra os estados de qualidade em tempo
real: verde indica condição OK, amarelo indica aviso ou ajuste recomendado e
vermelho indica problema, como rosto não detectado. A iluminação aparece em
amarelo enquanto seus limites finais ainda não estão calibrados. A mesma
legenda permanece visível durante a gravação.

### Menu interativo

Ao rodar `python main.py`, um menu é exibido:

```
1 - Capturar novo vídeo
2 - Analisar vídeo existente
3 - Capturar e analisar
4 - Benchmark (em breve)
5 - Configurações (em breve)
0 - Sair
```

### Argumentos de linha de comando

| Argumento | Descrição | Padrão |
|---|---|---|
| `--video <caminho>` | Analisa um vídeo já existente em vez de capturar | — |
| `--duration <s>` | Duração da captura em segundos | `30.0` |
| `--camera <índice>` | Índice da câmera a ser usada | `0` |
| `--capture-only` | Apenas grava o vídeo, sem rodar a análise | `False` |

Vídeos capturados são salvos em `data/captures/`, com o nome `video_<data>_<hora>.mp4` (a pasta é criada automaticamente na primeira captura).

### Calibração experimental

O módulo independente reutiliza o pipeline normal e não altera algoritmos, ROIs ou pesos:

```bash
python -m rPPG.calibration --captures 10 --duration 20 --output-dir calibration
```

Após cada captura, informe a FC exibida no Apple Watch, ou pressione ENTER para ausente. `reference_hr_bpm` é uma referência externa observacional, não ground truth clínico: nunca influencia resultado, algoritmo ou pesos. A saída contém JSON por captura, `results.csv`, `summary.csv`, espectros CSV e `report.html`; registra FC final, ROI × algoritmo, erros quando há referência, métricas de sinal/iluminação e audit. O limiar `0.784` é somente de pixel brilhante, não um quality gate.

---

## Configuração

Todos os parâmetros do pipeline ficam centralizados em `config.py`:

```python
MODEL_PATH = "assets/face_landmarker.task"
DEBUG_COMPARE_ALGORITHMS = True   # liga/desliga os gráficos de diagnóstico

HR_LOW_HZ = 0.7    # 42 bpm — limite inferior aceito para a FC
HR_HIGH_HZ = 4.0   # 240 bpm — limite superior aceito para a FC

METHOD_WEIGHTS = {"chrom": 0.3, "pos": 0.3, "ica": 0.3, "green": 0.1}

ROI_WEIGHTS = {
    "testa": 0.36,
    "bochecha_esquerda": 0.27,
    "bochecha_direita": 0.27,
    "glabela": 0.10,
}

ROI_POINTS = { ... }  # índices dos landmarks do MediaPipe para cada ROI

# Capture Quality (valores iniciais experimentais)
MOVEMENT_THRESHOLD = 0.010
MOVEMENT_WINDOW_SIZE = 5
READY_STABLE_FRAMES = 15
FACE_MIN_WIDTH_RATIO = 0.25
FACE_MAX_WIDTH_RATIO = 0.70
FACE_MIN_HEIGHT_RATIO = 0.35
FACE_MAX_HEIGHT_RATIO = 0.85
FACE_MAX_CENTER_OFFSET_X = 0.15
FACE_MAX_CENTER_OFFSET_Y = 0.18

# Iluminação: RGB normalizado [0, 1]; métrica observacional de clipping
LIGHTING_BRIGHT_PIXEL_CHANNEL = 0.784
```

`MOVEMENT_THRESHOLD` é o maior valor aceito para a métrica de movimento;
`MOVEMENT_WINDOW_SIZE` define quantas medições recentes compõem a suavização;
e `READY_STABLE_FRAMES` permanece disponível para futuros gates de prontidão,
mas não bloqueia a captura atual. `FACE_MIN/MAX_WIDTH_RATIO` e
`FACE_MIN/MAX_HEIGHT_RATIO` são os limites inclusivos da largura/altura da
bounding box divididas pelas dimensões do frame. `FACE_MAX_CENTER_OFFSET_X/Y`
limitam o desvio absoluto do centro da face em relação a 50% do frame, também
normalizado por largura/altura. Todos os valores são iniciais e devem ser
calibrados observando a webcam.

Os pesos de algoritmos somam 1. Os pesos legados das três ROIs preservam a
proporção 4:3:3, escalada para 90% do total; os 10% restantes são a contribuição
experimental explícita da glabela. O combinador valida pesos finitos e não
negativos e normaliza apenas os componentes explicitamente marcados como válidos.
O detalhamento de rastreabilidade está em [docs/signal-pipeline.md](docs/signal-pipeline.md).

---

## Arquitetura do projeto

```
rPPG/
├── main.py                        # CLI e menu interativo
├── config.py                      # constantes e pesos do pipeline
├── requirements.txt
├── README.md
├── assets/
│   └── face_landmarker.task       # modelo de landmarks faciais (MediaPipe)
│
├── capture/                       # pipeline de captura (independente da análise)
│   ├── capture_video.py           #   orquestra preview, gravação e feedback
│   └── quality_check.py           #   qualidade de movimento e enquadramento
│
├── analysis/                      # pipeline de análise (independente da captura)
│   └── analyze_video.py           #   lê um .mp4 e roda o pipeline completo de rPPG
│
├── roi/
│   ├── face_detection.py          #   wrapper do MediaPipe FaceLandmarker
│   └── roi_extraction.py          #   máscara + média RGB de cada ROI
│
├── preprocessing/
│   ├── smoothing.py                #   suavização por média móvel
│   └── filters.py                  #   filtro passa-banda Butterworth
│
├── extractors/                     # algoritmos de extração do sinal rPPG
│   ├── chrom.py                    #   CHROM (De Haan & Jeanne, 2013)
│   ├── pos.py                      #   POS (Wang et al., 2017)
│   ├── green.py                    #   GREEN (Verkruysse et al., 2008)
│   └── combine.py                  #   fusão ponderada entre algoritmos e ROIs
│
├── biomarkers/
│   ├── heart_rate.py               #   frequência cardíaca via pico da FFT
│   ├── hrv.py                      #   SDNN, RMSSD, pNN50
│   ├── signal_metrics.py           #   SNR, concentração espectral, amplitude, energia
│   └── respiratory_rate.py         #   ainda não implementado
│
├── reports/
│   ├── report.py                   #   relatório final e benchmarks no console
│   └── plots.py                    #   gráficos de diagnóstico (modo debug)
│
└── utils/
    └── models.py                   #   dataclass AnalysisResult
```

## Capture Quality: fluxo e teste manual

Capture Quality monitora condições de captura e apresenta feedback em tempo
real; não seleciona frames para o pipeline rPPG. O fluxo permanece independente
da análise:

```text
PREVIEW -> ENTER -> CAPTURING -> MP4 -> analyze_video.py
```

`FaceDetector` continua sendo responsável apenas por fornecer os landmarks do
MediaPipe. Esses landmarks alimentam dois componentes independentes em
`capture/quality_check.py`:

- `MovementQualityChecker` compara landmarks correspondentes em frames
  consecutivos, usa a mediana dos deslocamentos e normaliza pela diagonal da
  bounding box facial. A mediana em janela temporal reduz jitter do detector.
- `FaceFramingQualityChecker` calcula a bounding box dos landmarks, sua largura
  e altura relativas ao frame, e o centro normalizado da caixa. O enquadramento
  é aprovado apenas se tamanho e posição estiverem ambos dentro dos thresholds.

Durante CAPTURING, a bounding box é desenhada em tela. O overlay exibe largura,
altura, centro X/Y, movimento e enquadramento. A prioridade das mensagens de
enquadramento é: `ROSTO NAO DETECTADO`, `APROXIME O ROSTO`, `AFASTE O ROSTO`,
`CENTRALIZE O ROSTO` e `ROSTO ENQUADRADO`. Assim, quando há mais de um problema,
o ajuste de tamanho é pedido antes de centralização.

Para validar manualmente:

1. Execute `python main.py --capture-only` e confirme o PREVIEW; movimente-se
   livremente e verifique que ainda não existe MP4.
2. Pressione ENTER e confirme início imediato da contagem e da gravação.
3. Afaste-se, aproxime-se demais e desloque-se para esquerda, direita, acima e
   abaixo para observar as mensagens de framing e os percentuais de debug.
4. Posicione o rosto no centro, dentro dos limites, e confirme `ROSTO ENQUADRADO`.
5. Movimente a cabeça e confirme o aviso de movimento; estabilize novamente.
6. Combine movimento e enquadramento inadequado e confirme que ambos os estados
   são exibidos.
7. Durante todos os casos, confirme que o contador segue e o MP4 não é pausado.
8. Ao terminar, verifique que o MP4 contém somente CAPTURING e pode ser analisado
   pelo pipeline existente.

### Limitações atuais

Os thresholds de tamanho e centralização são valores iniciais, não limites
fisiológicos, e precisam de calibração experimental por webcam. A análise mede
iluminação na bounding box facial: `bright_pixel_ratio` é a proporção de pixels
em que `max(R,G,B) >= 0.784`, após normalização para [0, 1]. O valor 0,784 é o
threshold experimental/recomendado adotado nesta versão; não há referência
bibliográfica para ele no repositório. A métrica é registrada no audit, mas não
rejeita frames porque uma regra de exclusão calibrada ainda não foi definida.
Movimento, pose, oclusão, blur e qualidade clínica do sinal continuam exigindo
validação experimental.

---

## Saída da análise

`analyze_video()` retorna um `AnalysisResult` com:

- `heart_rate` — frequência cardíaca estimada (bpm)
- `hrv` — dicionário com `SDNN_ms`, `RMSSD_ms`, `pNN50_%` e `n_batimentos_detectados`
- `respiratory_rate` — não implementado
- `signal_metrics` — `snr`, `spectral_concentration`, `fft_peak`, `amplitude`, `std`, `energy`
- `fps`, `duration`, `valid_frames` — informações sobre a captura
- `audit` — proveniência de frames, iluminação, ROIs, algoritmos, pesos efetivos, alinhamento e sinal final

O relatório é impresso automaticamente via `print_report()` ao rodar `main.py`.
O audit também mostra toda exclusão e seu motivo; nenhum benchmark, `max()` ou
"Highest ..." altera o sinal final. Quando `DEBUG_COMPARE_ALGORITHMS = True`
(ou `debug=True` em `combine_roi_and_methods`), são mostrados gráficos de
diagnóstico para CHROM, POS, GREEN e ICA.

### Testes

Execute a partir do diretório pai de `rPPG/`:

```bash
python -m unittest rPPG.test_lighting_quality rPPG.test_quality_check rPPG.test_signal_pipeline rPPG.test_heart_rate rPPG.test_live_capture_quality
```


## Referências

- de Haan, G., & Jeanne, V. (2013). *Robust pulse rate from chrominance-based rPPG*. IEEE Transactions on Biomedical Engineering.
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). *Algorithmic principles of remote PPG*. IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). *Remote plethysmographic imaging using ambient light*. Optics Express.
