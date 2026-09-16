"""CLI orchestration for the independent rPPG capture and analysis pipelines."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow ``python main.py`` when the current directory is the rPPG package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rPPG.analysis.analyze_video import analyze_video
from rPPG.capture.capture_video import capture_video
from rPPG.reports.report import print_report


def _capture_and_analyze(camera, duration, reference_hr=None):
    """Run the complete capture followed by analysis workflow."""
    video_path = capture_video(camera, duration)
    result = analyze_video(video_path, reference_hr=reference_hr)
    print_report(result)
    return result


def _interactive_menu():
    """Present the user-facing rPPG workflow menu."""
    while True:
        print("\n=================================")
        print("        rPPG Biomarker System")
        print("=================================")
        print("1 - Capturar novo vídeo")
        print("2 - Analisar vídeo existente")
        print("3 - Capturar e analisar")
        print("4 - Benchmark (em breve)")
        print("5 - Configurações (em breve)")
        print("0 - Sair")
        option = input("\nEscolha uma opção: ").strip()

        try:
            if option == "1":
                print(capture_video())
            elif option == "2":
                video_path = input("Digite o caminho do vídeo:\n> ").strip().strip('"')
                result = analyze_video(video_path)
                print_report(result)
            elif option == "3":
                _capture_and_analyze(camera=0, duration=30.0)
            elif option == "4":
                print("Benchmark interativo será disponibilizado em breve.")
            elif option == "5":
                print("Configurações interativas serão disponibilizadas em breve.")
            elif option == "0":
                return None
            else:
                print("Opção inválida. Escolha uma opção do menu.")
        except (FileNotFoundError, RuntimeError) as error:
            print(f"Erro: {error}")


def main(argv=None):
    """Run the interactive menu or preserve the existing CLI workflows."""
    parser = argparse.ArgumentParser(description="Medição de HR/HRV via rPPG")
    parser.add_argument("--video", help="Caminho para um vídeo existente a ser analisado")
    parser.add_argument("--duration", type=float, default=30.0, help="Duração da captura em segundos")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera")
    parser.add_argument("--capture-only", action="store_true", help="Salva o vídeo sem analisá-lo")
    parser.add_argument(
        "--reference-hr",
        type=float,
        help="HR externo opcional (bpm), usado somente para validação do relatório",
    )
    args = parser.parse_args(argv)

    if argv is None and len(sys.argv) == 1:
        return _interactive_menu()

    if args.capture_only:
        video_path = args.video or capture_video(args.camera, args.duration)
        print(video_path)
        return video_path
    if args.video:
        result = analyze_video(args.video, reference_hr=args.reference_hr)
        print_report(result)
        return result
    return _capture_and_analyze(args.camera, args.duration, reference_hr=args.reference_hr)


if __name__ == "__main__":
    main()
