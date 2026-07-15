"""Runner: ejecuta el script original _full_eval_task23.py sin editarlo.
Solo añade en runtime el alias task23.load_t23 -> task23.load_task23 (el script
original llama load_t23 pero task23.py define load_task23). sentencepiece ya
instalado arregla los tokenizers Longformer. Se ejecuta desde Trabajo_LNR."""
import os, sys, runpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task23
if not hasattr(task23, "load_t23"):
    task23.load_t23 = task23.load_task23
runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "_full_eval_task23.py"), run_name="__main__")
