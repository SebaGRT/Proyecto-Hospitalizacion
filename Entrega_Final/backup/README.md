# Backup — Entrega_Final

**Fecha de备份:** 2026-05-13  
**Generado por:** Script de backup automático

## Contenido

| Archivo/Directorio | Descripción |
|---|---|
| `aaaaaaaaaaaaaaaaa.html` | Archivo HTML auxiliar |
| `Informe_Cientifico_AmatHerreraRodriguez.pdf` | Informe científico en PDF |
| `Informe_Final_AmatHerreraRodriguez.docx` | Informe final en Word |
| `main_paper.tex` | Código fuente LaTeX del paper |
| `Proyecto_Final_AmatHerreraRodriguez.ipynb` | Notebook principal del proyecto |
| `README.md` | Documentación de la carpeta Entrega_Final |
| `Rúbricas/` | Rúbricas de evaluación (5 archivos) |
| `outputs/` | Outputs generados (14 gráficos + 13 inferenciales) |
| `notebook_checksum.md5` | Checksum MD5 del notebook original |

## Archivos .bak

Archivos `.bak` creados junto a cada original en `Entrega_Final/`:
- `aaaaaaaaaaaaaaaaa.html.bak`
- `Informe_Cientifico_AmatHerreraRodriguez.pdf.bak`
- `Informe_Final_AmatHerreraRodriguez.docx.bak`
- `main_paper.tex.bak`
- `Proyecto_Final_AmatHerreraRodriguez.ipynb.bak`
- `README.md.bak`

## Verificación

Para verificar que los backups son idénticos a los originales:

```bash
diff -rq Entrega_Final/ Entrega_Final/backup/ --exclude=backup --exclude="*.bak"
# Debería devolver solo las líneas de diff para notebook_checksum.md5 y backup/README.md
```
