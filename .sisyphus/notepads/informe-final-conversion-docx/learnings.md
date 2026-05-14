## Learnings - Backup Task

- Use `workdir` in bash tool correctly: when workdir is `Entrega_Final`, target paths like `backup/` resolve correctly (no `Entrega_Final/backup/` prefix needed)
- `aaaaaaaaaaaaaaaaa.html` has 17 'a' characters - easy to miscount; always verify filenames via `ls -la` before scripting with them
- When copying a README.md into backup dir, it overwrites the intended backup README; must replace with backup-specific content after copy
- Total backup size: ~4.6MB (notebook is ~4.5MB of that)
