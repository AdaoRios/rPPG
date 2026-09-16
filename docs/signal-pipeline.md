# Auditoria do pipeline de sinal rPPG

## Proveniência do sinal

```text
vídeo -> frame RGB -> Face Landmarker -> ROIs sincronizadas -> média RGB
      -> CHROM/POS/GREEN/ICA por ROI -> z-score e alinhamento de polaridade
      -> fusão ponderada dos algoritmos por ROI -> fusão ponderada das ROIs
      -> filtro passa-banda final único -> HR / HRV / métricas finais
```

`analysis/analyze_video.py` coleta frames; `roi/face_detection.py` fornece os
478 landmarks pixelizados; `roi/roi_extraction.py` calcula as médias RGB.
`extractors/combine.py` é o único ponto de fusão e devolve sinal combinado sem
filtro mais um audit compacto. `analyze_video.py` aplica o filtro final único.
O mesmo `filtered_signal` alimenta `compute_hr_fft`, `compute_hrv` e
`compute_signal_metrics`.

`AnalysisResult.audit` guarda contagens, pesos, comprimentos, alinhamentos,
origem e exclusões. Os blocos de console `FRAME COLLECTION AUDIT` e `SIGNAL
PIPELINE AUDIT` mostram o resumo sem imprimir arrays de amostras.

## Diagnóstico da implementação anterior

O HR final **não vinha de uma única ROI ou algoritmo**. Antes desta alteração,
`combine_roi_and_methods` já fazia, em cada ROI:

```text
0.30*CHROM + 0.30*POS + 0.10*GREEN + 0.30*ICA
```

e depois uma média ponderada das ROIs. O resultado chegava a
`analyze_video.py` e era usado para HR, HRV e métricas. Assim, a coincidência de
96,43 bpm com POS/bochecha esquerda não prova seleção daquele componente. Com
168 amostras a 30 fps, a resolução FFT é cerca de 10,71 bpm; sinais distintos
podem cair no mesmo bin.

Havia três problemas: `np.average` normalizava pesos implicitamente; o menor
comprimento era selecionado e as caudas eram removidas sem registro no resultado;
o sinal final era filtrado no combinador e novamente na análise; e um frame era
descartado se qualquer máscara de ROI falhasse, sem motivo.

O ICA usa `argmax` para escolher internamente um de seus três componentes cegos
pelo maior poder espectral normalizado. Isso faz parte do ICA; não escolhe ROI
nem algoritmo para a fusão final. Índice e critério agora entram no audit.
`max()` no benchmark só apresenta `Highest ...` e não afeta a fusão.

## Pesos e combinação

| Escopo | Componentes e pesos |
| --- | --- |
| Algoritmos | CHROM 0,30; POS 0,30; GREEN 0,10; ICA 0,30 |
| ROIs | TESTA 0,36; BOCHECHA_ESQUERDA 0,27; BOCHECHA_DIREITA 0,27; GLABELA 0,10 |

Ambos somam 1. A glabela recebe 10% experimental; os pesos legados preservam a
proporção 4:3:3, escalada para 90%. O combinador valida pesos finitos e não
negativos e calcula explicitamente `peso / soma_dos_pesos` antes de executar
`sum(peso * sinal)`.

Componentes inválidos são registrados com a exceção no audit e os pesos dos
componentes restantes são renormalizados. Sinais de comprimentos diferentes são
alinhados explicitamente ao prefixo temporal comum, com perda registrada. Isso
é necessário porque o CHROM existente pode produzir cauda menor que os demais.

## ROIs e GLABELA

As ROIs são polígonos de landmarks no `config.py`. A GLABELA usa
`[107, 9, 336, 168]`: bordas de sobrancelhas na linha central, ponto superior e
ponte nasal. Em vídeo validado os pontos foram `(288,186)`, `(308,186)`,
`(328,184)` e `(307,208)`, formando região entre sobrancelhas. Ela percorre a
mesma erosão, RGB, quatro algoritmos, benchmark e fusão das outras ROIs.

Para preservar alinhamento temporal, um frame só é retido quando todas as ROIs
configuradas produzem máscara de pelo menos 50 pixels. Cada falha é contada por
ROI no audit; não há remoção silenciosa nem "melhor ROI".

## Iluminação

`LightingQualityChecker` normaliza RGB `[0,255]` ou `[0,1]` para `[0,1]`.
`LIGHTING_BRIGHT_PIXEL_CHANNEL = 0.784` define pixel brilhante quando
`max(R,G,B) >= 0.784`; `bright_pixel_ratio` é sua fração na bounding box facial.
É o threshold experimental/recomendado desta versão. Não há referência
bibliográfica para ele no repositório; ela deve ser adicionada depois.

O limiar é observacional: não existe uma taxa de pixels brilhantes calibrada
para justificar descarte. O audit informa frames avaliados, indisponíveis e
rejeitados por iluminação (atualmente zero); nenhuma heurística foi inventada.

## Benchmarks, HR e HRV

ROI benchmark: sinal normalizado de cada par ROI/algoritmo, filtrado para a
métrica. Algorithm benchmark: cada algoritmo após fusão ponderada das ROIs.
Relatório final: fusão dos algoritmos por ROI e das ROIs, seguida por filtro
Butterworth único de 0,7--4,0 Hz (42--240 bpm). Os benchmarks comparam
componentes; não são o sinal de produção.

HR seleciona o maior pico FFT na banda e converte Hz em bpm. HRV usa picos do
mesmo sinal final filtrado, IBIs de 250--1500 ms, SDNN, RMSSD e pNN50. Janelas
menores que 60 s são sinalizadas como exploratórias; uma janela de cerca de 6 s
não é suficiente para interpretação clínica de HRV.

## Validação executada

```text
python -m unittest rPPG.test_lighting_quality rPPG.test_quality_check rPPG.test_signal_pipeline rPPG.test_heart_rate rPPG.test_live_capture_quality
```

O comando passou 50 testes comportamentais: fórmula de fusão, normalização,
pesos inválidos, comprimentos incompatíveis, propagação de todas as ROIs e
algoritmos, exclusão auditada de ICA, GLABELA, limiar de iluminação, resolução
FFT, conversão Hz--bpm, limites da banda, sinais curtos e feedback do preview.

Também foi executado `data/captures/video_2026-09-02_17-48-29.mp4`: 184 frames,
30 fps, 6,13 s. As quatro ROIs e os quatro algoritmos foram válidos, sem
rejeição por face, máscara ou iluminação. CHROM determinou 168 amostras comuns.
O sinal final ponderado produziu HR 96,43 bpm, SNR -1,79 dB, concentração 0,40 e
HRV marcada como exploratória. A coincidência de HR com alguns componentes vem
da resolução FFT, não de seleção silenciosa.

Nesta etapa, o mesmo vídeo foi executado com `--reference-hr 85` apenas para
testar a apresentação de validação: HR 96,43 bpm, erro absoluto 11,43 bpm,
168 amostras efetivas, 10,71 bpm por bin e relação pico primário/segundo pico
de 1,02. A relação próxima de 1 reforça que a decisão espectral é ambígua.
Também foi analisado `video_2026-08-26_15-46-19.mp4`: 122 frames válidos,
120 amostras efetivas, HR 75,00 bpm e resolução de 15 bpm. O valor 85 bpm
informado nessa execução foi demonstrativo, não uma referência simultânea desse
arquivo, portanto não é uma validação fisiológica. Os dois vídeos confirmam o
funcionamento do audit, mas não justificam mudança de pesos ou de seleção.

## Dependências

Nenhuma dependência foi alterada. Ambiente validado: Python 3.10.11, NumPy
2.2.6, SciPy 1.15.3, OpenCV 5.0.0.93, MediaPipe 1.0.0 e Matplotlib 3.10.9.
`requirements.txt` continua sem pins de versão; isso preserva o ambiente atual,
mas reprodutibilidade entre instalações futuras ainda exige uma política de
versionamento.

## Pendências conhecidas

- Adicionar a fonte externa do threshold experimental de iluminação 0,784 ou
  substituí-lo por valor calibrado e referenciado.
- Definir e validar uma regra de rejeição por iluminação antes de converter a
  métrica observacional em quality gate.
- Validar as métricas de HRV contra referência fisiológica em janelas longas;
  resultados de captura curta permanecem exploratórios.


## HR Validation

The production estimator uses the effective final signal length, not the number
of frames originally read. It applies a Hann window and selects the maximum FFT
magnitude in 0.7--4.0 Hz (42--240 bpm). There is no zero-padding or peak
interpolation. The audit and the final report now expose:

- samples, effective FPS, duration, frequency resolution and BPM resolution;
- selected FFT bin, frequency, HR, magnitude and power;
- second distinct local spectral peak, ratio to the primary peak and separation;
- cardiac-band and total spectral power; and
- the explicit selection rule.

For the reported validation capture, 274 frames at 30 FPS would produce 264
effective samples after the current CHROM alignment. Its resolution is
30 / 264 = 0.11364 Hz, or 6.82 bpm. A 109.09 bpm output is bin 16
(1.81818 Hz). A reference near 85 bpm (1.41667 Hz) lies between bins 12
(81.82 bpm) and 13 (88.64 bpm), so exact 85 bpm cannot be emitted by the
current unpadded, non-interpolated FFT.

This resolution does not by itself explain the 109.09 bpm result. The supplied
SNR (-1.05 dB), spectral concentration (0.44), and disagreement across ROI and
algorithm benchmarks indicate a weak/ambiguous spectral decision. It is not
possible to label the 109.09 bpm peak as a harmonic or artefact from the
reported summary alone: the saved spectrum's second peak and its amplitude are
required. The new audit records that evidence for subsequent captures.

Synthetic tests establish that 1.5 Hz produces 90 bpm and 1.41667 Hz produces
85 bpm on suitable FFT-aligned windows. A separate test demonstrates that the
documented maximum-magnitude rule can select a dominant 3 Hz harmonic over a
1.5 Hz fundamental. This is a limitation now made visible, not a reference-led
change to the production HR.

An optional --reference-hr <bpm> CLI argument adds Reference HR, absolute error
and relative error to the report. It only annotates benchmarks and the final
result; it cannot change weights, peak selection, ROIs, or the final signal.

## Live Capture Quality

The OpenCV preview and recording overlay now displays, on every frame:

- FACE: OK when Face Landmarker returns landmarks;
- ROIs: valid/4, using the same ROI-mask validation as offline analysis;
- lighting measurement state and the current BRIGHT PIXEL RATIO (>= 0.784);
- mean luminance, dark-pixel ratio and illumination uniformity; and
- existing framing and movement feedback.

The live checks remain observational. LIGHTING_BRIGHT_PIXEL_CHANNEL = 0.784
still defines only whether a pixel contributes to bright-pixel ratio after RGB
normalization to [0,1]. No ratio threshold exists to honestly label illumination
as approved, marginal, or insufficient, so the UI states that no calibrated
quality gate exists rather than misleadingly reporting OK. Missing face or
ROIs is shown as an error. The additional work is four small ROI masks after
the already-required Face Landmarker call and does not affect which frames the
capture writer records.

## Current Limitations

- An approximately 9-second capture has about 6.6--6.8 bpm FFT-bin resolution,
  depending on effective aligned samples; it is not a fine-resolution HR
  estimate.
- Harmonic and spurious-peak differentiation is diagnostic-only. It needs
  multi-capture validation and reference data before changing production peak
  selection.
- External reference integration is optional and not simultaneous by design;
  it is only a validation label.
- HRV remains exploratory for short windows.
- Lighting monitoring has no calibrated pass/fail ratio or rejection gate.
- Multiple captures with the same protocol are required before any weight or
  algorithm policy change.
