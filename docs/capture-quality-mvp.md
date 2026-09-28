# Capture Quality MVP — auditoria técnica

> Base: branch `capture-quality-mvp`, `HEAD` `5a4918e` (2026-08-19).
> Auditoria feita em 2026-08-23. As etiquetas distinguem código verificado de
> interpretação e trabalho futuro.

## 1. Objetivo

O Capture Quality atual monitora **movimento** e **enquadramento facial**
durante a webcam. Ele fornece feedback no overlay OpenCV; não seleciona frames,
não calcula score agregado e não decide se biometria pode começar.

- **[IMPLEMENTADO]** movimento e enquadramento durante preview e gravação.
- **[IMPLEMENTADO]** métricas observacionais de iluminação facial por frame.
- **[NÃO IMPLEMENTADO]** thresholds finais de iluminação/exposição, blur, pose,
  oclusão, score geral e quality gate.
- **[PARCIAL]** `consecutive_stable_frames` é calculado, mas não libera captura.

## 2. Estado atual e arquitetura

| Arquivo | Papel atual |
|---|---|
| `main.py` | CLI/menu; orquestra captura e/ou análise. |
| `capture/capture_video.py` | Loop OpenCV, preview, MP4 e overlay. |
| `capture/quality_check.py` | Checkers/dataclasses de movimento e enquadramento. |
| `capture/lighting_quality.py` | Checker/dataclass de métricas de iluminação facial. |
| `roi/face_detection.py` | Wrapper MediaPipe Face Landmarker em modo `VIDEO`. |
| `config.py` | Modelo, thresholds experimentais e janela. |
| `analysis/analyze_video.py` | Pipeline offline do MP4 para rPPG. |
| `roi/roi_extraction.py` | Médias RGB de ROIs de testa e bochechas. |
| `extractors/*`, `preprocessing/*`, `biomarkers/*` | Fusão rPPG, filtros e métricas. |
| `test_quality_check.py` / `test_lighting_quality.py` | Testes unitários de framing/movimento e iluminação. |

Dependências: `opencv-python` (câmera/MP4/UI), `mediapipe` (landmarks),
`numpy`, `scipy` (filtros/FFT) e `matplotlib` (diagnóstico opcional).

### Fluxo implementado

```text
main.main()
  -> _capture_and_analyze(camera, duration) [exceto --capture-only]
    -> capture_video(camera_index, duration_s)
      -> cv2.VideoCapture(camera_index).read() [frame BGR]
      -> cv2.cvtColor(..., BGR2RGB)
      -> FaceDetector.detect(rgb_frame, timestamp_ms)
      -> MovementQualityChecker.update(landmarks)
      -> FaceFramingQualityChecker.evaluate(landmarks, frame.shape)
      -> LightingQualityChecker.evaluate(rgb_frame, framing.bbox)
      -> overlay + cv2.imshow()
      -> (depois de ENTER) VideoWriter.write(frame) -> data/captures/*.mp4
    -> analyze_video(video_path)
      -> detect() + extract_roi_means() por frame válido
      -> combine_roi_and_methods() -> bandpass_filter()
      -> AnalysisResult e print_report()
```

`capture_video()` e `analyze_video()` são independentes. A análise redetecta
landmarks no MP4 e não consulta os resultados de Capture Quality.

## 3. Quality

Há dois sentidos diferentes de "quality": qualidade de captura por landmarks e
métricas do sinal rPPG produzidas somente após a análise offline.

### Movimento — `MovementQualityChecker`

**[IMPLEMENTADO]** em `capture/quality_check.py`.

| Item | Contrato/semântica |
|---|---|
| Entrada | `landmarks`: `None` ou sequência conversível em matriz NumPy `N×2`/`N×>=2`, em pixels; colunas após x,y são ignoradas. O detector entrega lista de pares `int`. |
| Estado | Landmark anterior, `deque(maxlen=MOVEMENT_WINDOW_SIZE)` e contador consecutivo. |
| Cálculo por frame | `d_i=||p_i(t)-p_i(t-1)||`; `D=||max(p(t))-min(p(t))||`; `m_t=mediana(d_i)/D`. |
| Janela | `movement_metric=mediana` dos últimos até 5 `m_t`; é temporal, não de vídeo completo. |
| Saída | `MovementQualityResult(movement_metric, movement_threshold, is_stable, consecutive_stable_frames)`. |
| Unidade/faixa | Razão adimensional não negativa; não há máximo definido. É `None` no primeiro frame após reset, face ausente, shape alterado ou diagonal nula. |
| Interpretação | `is_stable` é `movement_metric <= 0.010`; o primeiro frame não é estável. |

`update(None)` faz `reset()`. Entrada inválida não-`N×2` lança `ValueError`.

### Enquadramento — `FaceFramingQualityChecker`

**[IMPLEMENTADO]** em `capture/quality_check.py`; é stateless e por frame.

| Item | Contrato/semântica |
|---|---|
| Entrada | landmarks em pixels (ou `None`) e `frame_shape` com `(altura, largura)`. |
| Processamento | Converte para `float64`, rejeita inválidos/não finitos, calcula bbox e limita aos limites do frame. |
| Métricas | `face_width_ratio=(xmax-xmin)/largura`; `face_height_ratio=(ymax-ymin)/altura`; centros são o ponto médio dividido por largura/altura. |
| Saída | `FaceFramingQualityResult` abaixo. |
| Unidade/faixa | Razões adimensionais entre 0 e 1 após clipping. |
| Decisão | `size_ok` exige largura/altura nos intervalos inclusivos; `position_ok` exige centro próximo de 0,5; `framing_ok=size_ok and position_ok`. |

```python
{
  "face_detected": bool,
  "face_width_ratio": float | None,
  "face_height_ratio": float | None,
  "face_center_x": float | None,
  "face_center_y": float | None,
  "size_ok": bool,
  "position_ok": bool,
  "framing_ok": bool,
  "bbox": tuple[int, int, int, int] | None,
  "message": str,
}
```

Exemplo coberto pelo teste: `[[350,300],[650,700]]` em `(1000,1000,3)` retorna
razões `0.3`/`0.4`, centros `0.5`/`0.5`, flags verdadeiros e
`ROSTO ENQUADRADO`. Sem face/entrada inválida, métricas e bbox são `None`,
flags são falsos e a mensagem é `ROSTO NAO DETECTADO`.

Prioridade de mensagem: face ausente; pequena (`APROXIME O ROSTO`); grande
(`AFASTE O ROSTO`); descentralizada (`CENTRALIZE O ROSTO`); adequada
(`ROSTO ENQUADRADO`).

### Calibração experimental de Face Framing

**[IMPLEMENTADO]** Os limites de tamanho foram recalibrados exclusivamente a
partir de testes manuais com uma webcam e um usuário. As observações foram:

| Largura / altura da face | Observação manual |
|---|---|
| 20,5% / 33,3% | Pequena demais. |
| 26,4% / 44,0% | Confortável, mas já com poucos pixels para ROIs. |
| 32,0% / 55,4% | Referência de captura muito boa. |
| 36,4% / 65,1% | Confortável e com pixels adequados. |
| 44,4% / 77,3% | Menos confortável, porém ainda dentro do frame. |
| 52,2% / 90,0% | Próxima demais, com pouca margem de segurança. |

Também foi observado que, perto de 45% de largura, pequenos movimentos não
tiraram o rosto do frame. Por isso os valores abaixo são uma calibração inicial
do MVP: **não são clinicamente nem universalmente validados**.

Tamanho e posição continuam decisões independentes: `size_ok` exige as duas
razões dentro da faixa, `position_ok` exige ambos os centros dentro da faixa, e
somente `framing_ok = size_ok and position_ok` reúne as duas condições. O
`FaceFramingQualityResult` mantém seus campos estruturados e o checker não
depende de OpenCV/UI; o overlay apenas consome o seu resultado. As comparações
são inclusivas e usam somente uma tolerância de precisão de ponto flutuante
para não rejeitar um valor matematicamente igual ao limite.

### Métricas de qualidade do sinal rPPG

**[IMPLEMENTADO, mas separado do Capture Quality]**
`biomarkers/signal_metrics.py:compute_signal_metrics(filtered_signal, fps)`
recebe vetor NumPy 1-D já filtrado e FPS, e retorna:

| Campo | Cálculo/unidade |
|---|---|
| `std` | Desvio-padrão; unidade do sinal. |
| `amplitude` | `max-min`; unidade do sinal. |
| `energy` | Soma dos quadrados; unidade ao quadrado. |
| `fft_peak` | Maior magnitude FFT normalizada na banda 0,7–4,0 Hz. |
| `snr` | `10 log10(potência_pico/potência_ruído)`, em dB; pode ser `inf`. |
| `peak_ratio` | Pico FFT / média do ruído; pode ser `inf`. |
| `spectral_concentration` | Potência de cinco bins centrados no pico / total da banda; [0,1] quando há potência. |

Uma cópia de média zero/desvio um é usada no cálculo espectral. Não há score
composto, threshold de aceitação nem ligação desse dicionário ao overlay. Essas
métricas são de vídeo completo (e benchmarks por ROI/método), nunca por frame.

### Thresholds atuais

| Constante/local | Valor | Unidade/efeito atual |
|---|---:|---|
| `MOVEMENT_THRESHOLD` (`config.py`) | `0.010` | Razão; define `is_stable`. |
| `MOVEMENT_WINDOW_SIZE` | `5` | Medições na janela mediana; não é threshold. |
| `READY_STABLE_FRAMES` | `15` | Frames consecutivos; calculado, sem gate. |
| `FACE_MIN_WIDTH_RATIO` / `MAX` | `0.28` / `0.45` | Razão de largura; `size_ok`; calibração manual experimental. |
| `FACE_MIN_HEIGHT_RATIO` / `MAX` | `0.45` / `0.80` | Razão de altura; `size_ok`; calibração manual experimental. |
| `FACE_MAX_CENTER_OFFSET_X` / `Y` | `0.15` / `0.18` | Desvio normalizado a 0,5; `position_ok`. |
| `FaceDetector` confidences | `0.6` cada | Detecção, presença e tracking do MediaPipe. |

Os thresholds de Face Framing acima são experimentais e específicos à calibração
manual descrita; exigem nova avaliação para outros dispositivos, pessoas e
condições. **[NÃO IMPLEMENTADO]** threshold de iluminação, blur, qualidade
geral, score de prontidão ou mínimo de frames para iniciar biometria. O filtro
cardíaco 0,7–4,0 Hz é processamento rPPG, não threshold de qualidade de captura.

## Lighting Quality

### Objetivo e escopo

**[IMPLEMENTADO]** `capture/lighting_quality.py` mede, por frame, a iluminação
da região facial já delimitada pela `bbox` do Face Framing. O checker recebe um
frame RGB e `face_bbox=(x_min, y_min, x_max, y_max)` em pixels, recorta somente
essa região e devolve `LightingQualityResult`. Assim, fundos claros ou escuros
não entram no cálculo. Não há nova detecção facial, ROI anatômica, melhoria de
imagem ou alteração no pipeline rPPG.

O resultado estruturado contém `face_detected`, `mean_luminance`,
`dark_pixel_ratio`, `bright_pixel_ratio`, `illumination_uniformity`,
`lighting_ok`, `status` e `message`. Com face/região indisponível, as métricas
são `None` e `status` diferencia `FACE_NOT_DETECTED` de `INVALID_INPUT`.
Quando há métricas, `status` é `METRICS_AVAILABLE` e `lighting_ok` é sempre
`None`: o projeto ainda não toma uma decisão de qualidade de iluminação.

### Métricas e cálculo

O checker normaliza entradas RGB finitas de `[0,255]` ou `[0,1]` para `[0,1]`.
Para uma região com pixels `R'`, `G'`, `B'` normalizados, a luma relativa usada
como luminância operacional é a transformação Rec. 709:

```text
Y' = 0.2126 R' + 0.7152 G' + 0.0722 B'
```

| Métrica | Definição | Faixa |
|---|---|---|
| `mean_luminance` | Média espacial de `Y'` na bbox facial. | [0,1] |
| `dark_pixel_ratio` | Fração de pixels cujo `Y' <= LIGHTING_DARK_PIXEL_LUMINANCE`. | [0,1] |
| `bright_pixel_ratio` | Fração de pixels cujo maior canal RGB é `>= LIGHTING_BRIGHT_PIXEL_CHANNEL`; proxy de canal muito claro/próximo de clipping. | [0,1] |
| `illumination_uniformity` | Divide a luma em uma grade configurável. Se `u_i` são as médias das células, retorna `1 - (max(u_i)-min(u_i))`, limitado a [0,1]. | [0,1]; maior é mais uniforme |

A uniformidade usa distribuição espacial, não apenas média global: por exemplo,
uma metade preta e outra branca em uma grade 2×2 gera uniformidade 0, enquanto
uma região de luma constante gera 1.

### Parâmetros e estado de calibração

| Configuração | Valor inicial | Papel |
|---|---:|---|
| `LIGHTING_LUMA_WEIGHTS` | `(0.2126, 0.7152, 0.0722)` | Pesos Rec. 709 de `Y'`. |
| `LIGHTING_DARK_PIXEL_LUMINANCE` | `0.10` | Classificação de pixel escuro para métrica. |
| `LIGHTING_BRIGHT_PIXEL_CHANNEL` | `0.784` | Classificação de canal muito claro para métrica. |
| `LIGHTING_UNIFORMITY_GRID_ROWS/COLUMNS` | `2` / `2` | Resolução espacial da métrica de uniformidade. |

Os dois valores de classificação de pixel **não são thresholds finais de
qualidade** e não definem `lighting_ok`. **[NÃO IMPLEMENTADO] Os thresholds
finais de iluminação ainda não foram calibrados.**

### Relação com a literatura e limitações

O estudo de Li et al. avaliou rPPG em sete níveis de iluminância (6,3 a 100,0
lux) e relata que a intensidade e o tipo de luz influenciam a qualidade de HR,
além de diferenças entre ROIs faciais. Ele motiva medir a iluminação e manter a
seleção de ROI separada, mas seus valores de lux são referências experimentais:
**não podem ser convertidos diretamente em thresholds de pixel de webcam**.
[Artigo original](https://www.nature.com/articles/s44325-026-00140-7).

**[PARCIAL]** As métricas atuais são proxies de imagem e não medem iluminância
física, espectro da fonte, exposição da câmera, tom de pele ou qualidade rPPG.
Também não adicionam glabella, nasal dorsum ou malar; essas ROIs seguem como
trabalho posterior.

### Overlay

`capture_video()` calcula as métricas a cada frame e exibe `Lighting`, `Mean
luminance`, `Dark ratio`, `Bright ratio` e `Uniformity` no overlay, tanto no
preview quanto durante a gravação. O overlay somente observa valores; ele não
usa cores, mensagens de correção ou gate de captura para iluminação.

## 4. Camera check

Não existe função/classe chamada literalmente `camera check`. **[INFERÊNCIA]**
o termo corresponde à composição dos checkers no loop de `capture_video()` e
do `FaceDetector`.

### Condição: face detectada

- **Entrada:** RGB `H×W×3` e timestamp ms crescente.
- **Processamento/métrica:** `FaceLandmarker.detect_for_video`, no máximo uma
  face, confidences 0,6.
- **Saída:** lista de `(x,y)` pixels ou `None`.
- **Threshold:** três confidences 0,6.
- **Implementado:** **SIM**; pose, oclusão e segmentação não existem.

### Condição: pouco movimento/estabilidade

- **Entrada:** landmarks correspondentes de frames consecutivos.
- **Processamento/métrica:** mediana de deslocamentos/diagonal e mediana de até
  cinco valores recentes.
- **Saída:** `MovementQualityResult`; overlay mostra métrica, threshold e
  `OK`/`MOVIMENTO DETECTADO` durante CAPTURING.
- **Threshold:** `<=0.010`; contador de 15 não altera captura.
- **Implementado:** **SIM**, com estado temporal; nenhum frame é recusado.

### Condição: tamanho e centralização

- **Entrada:** landmarks e dimensões do frame.
- **Processamento/métrica:** bbox, razões de largura/altura/centro.
- **Saída:** `FaceFramingQualityResult`, retângulo e mensagem de overlay.
- **Threshold:** seis valores `FACE_*` acima.
- **Implementado:** **SIM**, por frame; não há estabilidade temporal.

### Condição: iluminação

- **Entrada:** frame RGB e `framing.bbox` derivada dos landmarks existentes.
- **Processamento/métricas:** luma Rec. 709 média, proporções de pixels
  escuros/muito claros e uniformidade espacial por grade 2×2.
- **Saída:** `LightingQualityResult`; métricas normalizadas e `lighting_ok=None`.
- **Threshold:** classificadores de pixel configuráveis, não thresholds de
  qualidade finais.
- **Implementado:** **PARCIAL**; mede e exibe dados, mas não classifica a
  iluminação como adequada/inadequada nem bloqueia frames.

`capture_video()` só usa resultados para texto/cor/bbox: não retorna resultado
de sessão, callback, JSON nem saída por frame.

## 5. Interface interna atual

| Função/classe | Arquivo | Entrada | Saída | Finalidade |
|---|---|---|---|---|
| `FaceDetector.detect` | `roi/face_detection.py` | RGB `H×W×3`, timestamp ms | lista `(x,y)` ou `None` | Landmarks MediaPipe. |
| `MovementQualityChecker.update` | `capture/quality_check.py` | landmarks ou `None` | `MovementQualityResult` | Movimento temporal. |
| `FaceFramingQualityChecker.evaluate` | mesmo | landmarks, `frame.shape` | `FaceFramingQualityResult` | Tamanho/posição por frame. |
| `LightingQualityChecker.evaluate` | `capture/lighting_quality.py` | frame RGB, bbox facial | `LightingQualityResult` | Métricas de iluminação por frame. |
| `capture_video` | `capture/capture_video.py` | índice OpenCV, duração, diretório opcional | caminho `str` ou erro | Preview e gravação. |
| `analyze_video` | `analysis/analyze_video.py` | caminho `str`/`Path` | `AnalysisResult` | Análise offline. |
| `compute_signal_metrics` | `biomarkers/signal_metrics.py` | vetor filtrado, FPS | dict de sete métricas | Qualidade de sinal. |

`AnalysisResult` contém `heart_rate` (bpm), `hrv` (dict),
`respiratory_rate` (`None`), `signal_metrics`, `fps`, `duration` e
`valid_frames`.

## 6. Captura de vídeo

### Vídeo gravado

**[IMPLEMENTADO]** `analyze_video()` abre o arquivo inteiro, acumula médias RGB
dos frames com face e ROIs válidas e só então executa fusão, FFT e HRV. Há uma
guarda explícita de dois frames válidos; os algoritmos/filtros têm requisitos de
amostra implícitos maiores, sem uma validação prévia dedicada.

### Câmera em tempo real

**[PARCIAL]** Aquisição, Face Landmarker, movimento e framing são frame a frame
no mesmo loop. Movimento tem buffer de no máximo cinco valores; framing não tem
buffer. Depois de ENTER, todos os frames são gravados, mesmo quando os checkers
apontam problema.

**[NÃO IMPLEMENTADO]** Biometria incremental, buffer de ROIs em janelas, HR ou
qualidade de sinal contínua e entrega stream/por frame. Logo, há tempo real
para feedback de captura, não para rPPG/biomarcadores, que ocorrem após salvar
o MP4. A latência não é medida; MediaPipe e renderização são síncronos.

## 7. CLI / `main.py`

Argumentos implementados: `--video PATH`, `--duration FLOAT` (padrão 30.0),
`--camera INT` (padrão 0) e `--capture-only`. Sem argumentos abre menu;
`--video` analisa somente; com `--capture-only` grava somente; nos demais casos
captura e depois analisa.

### `python main.py --duration 20 --camera 1`

O comando foi executado com o Python da `.venv`:

```text
.\.venv\Scripts\python.exe main.py --duration 20 --camera 1
```

Ele é válido: `argparse` aceita os argumentos e o fluxo é
`main -> _capture_and_analyze(1,20.0) -> capture_video(1,20.0) ->
cv2.VideoCapture(1)`. Nesta máquina, `isOpened()` foi falso e a aplicação
produziu:

```text
RuntimeError: Nao foi possivel acessar a camera.
```

Portanto, `1` é passado sem conversão como índice OpenCV: não há enumeração,
nome de dispositivo ou fallback para `0`. A causa observada é câmera de índice
1 inacessível neste ambiente, antes de preview, detector ou MP4. Isso não prova
que o índice seja inválido em outra máquina: depende dos dispositivos e das
permissões locais.

Nesta sessão, o executável resolvido por `python` não pôde iniciar; a `.venv`
(Python 3.14.3) permitiu CLI e testes. É uma limitação observada de ambiente ou
launcher, não do parser do projeto.

Com uma câmera acessível, esse comando não inicia os 20 s automaticamente:
abre PREVIEW e exige ENTER. A duração começa após ENTER; `Q` encerra. Menos de
dois frames remove somente o MP4 recém-criado e lança `RuntimeError`.

## 8. Possível integração futura com API

Reutilizáveis: `FaceDetector.detect` (com ciclo de vida explícito),
`MovementQualityChecker.update`, `FaceFramingQualityChecker.evaluate` e, após
encerrar um arquivo, `analyze_video`. Os checkers não dependem do console.
`capture_video` está acoplado à webcam, UI, teclado e MP4; `analyze_video`
depende de arquivo e `combine_roi_and_methods` imprime benchmarks.

**[FUTURO]** Uma resposta por frame baseada apenas no que já existe poderia ser:

```json
{
  "face_detected": true,
  "movement_metric": 0.0042,
  "is_stable": true,
  "consecutive_stable_frames": 4,
  "face_width_ratio": 0.31,
  "face_height_ratio": 0.42,
  "face_center_x": 0.50,
  "face_center_y": 0.49,
  "framing_ok": true,
  "framing_message": "ROSTO ENQUADRADO"
}
```

Seria necessário adaptar posse dos objetos por sessão, validar/converter frames
e serializar dataclasses. **[PARCIAL]** iluminação já fornece métricas, mas não
um julgamento calibrado; `quality_score`, status de aptidão e biometria contínua
continuam **[NÃO IMPLEMENTADOS]**.

## 9. Thresholds e decisão futura

| Condição | Métrica existente | Tempo real/estabilidade | Threshold atual | Lacuna |
|---|---|---|---|---|
| Movimento | razão de deslocamento mediano | Sim; janela de 5 e contador consecutivo | 0,010 | Calibrar e definir uso dos 15 frames. |
| Posicionamento | tamanho e centro normalizados da bbox | Sim; somente frame atual | Tamanho: 28%–45% largura, 45%–80% altura; centro: X 35%–65%, Y 32%–68% | Calibrar em outros cenários; possível estabilidade temporal. |
| Iluminação | Luma média, ratios escuro/claro e uniformidade espacial | Sim; por frame, sem estado temporal | Só classificadores de pixels; nenhum limite de qualidade | Calibrar em webcam/ambientes reais e definir decisão. |
| Qualidade geral | Métricas espectrais rPPG offline | Não durante captura | Nenhum score | Definir score, janelas e validação. |

Movimento e framing podem virar estados binários, mas seus valores são
experimentais. Métricas espectrais poderiam informar qualidade geral, mas só
existem depois de acumular/processar o vídeo completo.

## 10. Testes

Existem `test_quality_check.py` e `test_lighting_quality.py`; ainda não há
testes versionados para câmera, OpenCV, `FaceDetector`, MP4, `analyze_video`,
ROI anatômica ou métricas de sinal.

Execução realizada, sem alterar testes:

```text
.\.venv\Scripts\python.exe -m unittest discover -v
Ran 29 tests — OK
```

Os testes de Face Framing cobrem tamanho abaixo/no limite mínimo/dentro/no
limite máximo/acima do máximo, os quatro limites inclusivos de centralização,
rosto fora da região, ausência de face e entradas inválidas/degeneradas. Os
testes de iluminação cobrem frame uniforme/escuro/claro, ratios de pixels na
ROI, isolamento do fundo, uniformidade alta/baixa, ausência de face, entradas
inválidas e intervalos normalizados. Não exercitam hardware, MediaPipe, overlay,
gravação nem rPPG.

## 11. Histórico relevante

Há um único commit em `main..capture-quality-mvp`:

| Commit/data | Arquivos | Mudança e impacto |
|---|---|---|
| `5a4918e` — 2026-08-19 18:05:47 -03:00, **Add detecção movimento e face framing (precisa aprimorar)** | `capture/capture_video.py`, `capture/quality_check.py`, `config.py`, `README.md`, novo `test_quality_check.py` | Substituiu aceitação por face+brilho por preview/gravação incondicional com feedback de movimento/framing; adicionou parâmetros e testes. |

Antes do commit, `is_frame_quality_acceptable(rgb_frame, landmarks)` aceitava
apenas face detectada e brilho médio RGB entre **20,0 e 235,0**. A função e
esses thresholds não estão no `HEAD`. O `LightingQualityChecker` atual é uma
implementação nova de métricas estruturadas; ele não reintroduz esse gate antigo.

## 12. Limitações conhecidas

- Capture Quality não retorna resultado estruturado da sessão nem persiste
  métricas junto ao MP4.
- Não há gate: após ENTER, face ausente, movimento ou framing inadequado ainda
  gravam todos os frames.
- `READY_STABLE_FRAMES` não participa de decisão.
- Iluminação possui métricas observacionais, mas não thresholds calibrados,
  decisão `lighting_ok`, medição em lux ou validação de qualidade biométrica.
- Exposição da câmera, blur, pose e oclusão continuam fora do escopo.
- A análise rPPG usa vídeo completo e imprime benchmarks no console; não é API.
- Não há evidência no código de calibração com dados reais.

## 13. Próximos passos

### Já implementado

Landmarks MediaPipe por frame, feedback de movimento/framing, gravação contínua
pós-ENTER, análise offline e Face Framing calibrado experimentalmente para a
webcam/usuário testados. Os testes incluem os limites inclusivos de tamanho e
centralização. Lighting Quality mede e exibe métricas estruturadas por região
facial, sem decisão final.

### Parcialmente implementado

Prontidão temporal: contador existe sem efeito operacional. Captura em tempo
real: feedback sim; biometria/sinal não.

### Ainda não implementado

Thresholds finais e decisão de iluminação, demais condições citadas, score,
gate de captura/processamento, resultado serializável por sessão e rPPG
incremental. As métricas de qualidade do sinal rPPG continuam separadas do
Capture Quality.

### Sugestões futuras

1. Calibração experimental dos thresholds utilizando diferentes condições reais de iluminação.
2. Reavaliar/calibrar Face Framing em dispositivos, usuários e ambientes diversos.
3. Definir contrato de sessão/API e separar captura/UI da avaliação pura.
4. Estabelecer janela mínima e validações para qualidade do sinal rPPG, que permanece separado.
5. Cobrir câmera, MediaPipe, MP4 e pipeline offline com testes de integração.
