import nbformat, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'c:\Users\vjrrg\Documents\UDD\Proyecto-Hospitalizacion\Avance 2\Avance2_Proyecto_Final_updated.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Check cells 27, 29, 34 (old analysis text that wasn't updated), 
# 45 (Escenario A old), 61 (confusion matrix old), 62 (old predictive), 64 (old OR), 72 (old OLS predictive)
# Also check last cell (74) for references
old_markers = {
    27: 'brecha de cinco veces',  # Old mortalidad barplot text
    29: 'supera los 5 procedimientos',  # Old procedimientos barplot text  
    34: 'Coherencia con la Literatura',  # Old Table 3 analysis - very long
    45: 'Si el OR es significativo',  # Old Escenario A - generic
    62: 'Si el AUC es modesto',  # Old predictive interpretation - generic
    64: 'El forest plot muestra',  # Old OR interpretation
    72: 'Un R² moderado (~0.20–0.40) es esperable',  # Old OLS predictive - wrong range per report
}

print('=== CELLS WITH OLD/GENERIC TEXT STILL PRESENT ===')
for idx, marker in old_markers.items():
    if idx < len(nb.cells) and nb.cells[idx].cell_type == 'markdown':
        if marker in nb.cells[idx].source:
            print(f'  CELL [{idx}]: Still contains old text: "{marker}"')
            print(f'    First 150 chars: {nb.cells[idx].source[:150]}')
            print()

# Check cell 74 has references section
cell74 = nb.cells[74].source if len(nb.cells) > 74 else ''
has_refs = 'Referencias' in cell74 or 'Wennberg' in cell74
has_next_steps = 'Próximos Pasos' in cell74
has_discussion = 'Discusión' in cell74
print(f'\n=== CELL [74] FINAL SECTION CHECK ===')
print(f'  Has Discussion: {has_discussion}')
print(f'  Has Next Steps: {has_next_steps}')
print(f'  Has References: {has_refs}')
print(f'  Length: {len(cell74)} chars')

# Check if section numbers are consistent
print('\n=== SECTION NUMBER AUDIT ===')
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        for line in cell.source.split('\n'):
            if line.strip().startswith('# ') and not line.strip().startswith('##'):
                print(f'  [{i:2d}] {line.strip()[:80]}')
