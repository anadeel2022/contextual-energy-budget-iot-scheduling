V22 plotting-only replacement for V21 long-run simulations

1) Copy make_figures_v22.py into the Sensors_V21_Final_LongRun directory.
2) Rename the copy to make_figures.py, replacing the original file.
3) From that directory in Windows PowerShell, execute:
   py -3.13 .\make_figures.py --out outputs\final_200

Alternatively, if you do not want to replace the original file:
   py -3.13 .\make_figures_v22.py --out outputs\final_200

For a completed 500-seed run, use --out outputs\final_500.

Reads completed experiment files only:
  episodes.csv, summary.csv, paired_ci.csv, protocol.json
Writes six pairs of figures (PDF + 250-dpi PNG) under:
  <output>/figures/Fig02_... through Fig07_...

Does not rerun simulations, change policy decisions, or change numerical results.
Figure 4: true centered figure legend with reserved space above titles and controlled x ticks.
Figure 5: centered legend, improved x-label clearance.
Other figures: consistent plot styling, legend spacing and clearer labels.

Tested using the archived 20-seed V21 independent evaluation files; all six PDFs generated successfully.
