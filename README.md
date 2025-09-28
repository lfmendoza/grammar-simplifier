# Grammar Simplifier

Solución de laboratorio para simplificación de gramáticas y conversión a CNF.

## Requisitos

- Python 3.10+
- Solo librerías estándar (sin dependencias externas).

## Formato de entrada

- No-terminales: `A..Z`
- Terminales: `a..z` o `0..9`
- Epsilon: `ε` o `eps`
- Flecha: `->`
- OR: `|`
- Una producción por línea:

```
S -> 0A0 | 1B1 | BB
A -> C
B -> S | A
C -> S | ε
```

## Uso

```bash
python3 grammar_simplifier.py grammars/g1.txt
python3 grammar_simplifier.py grammars/g2.txt --no-steps
```

## Procedimiento

Ver docs/procedimiento.pdf (incluye validación por regex, eliminación de ε, unitarias, símbolos inútiles, y CNF).

---

# Indicaciones finales para el **video** (Problema 1)

1. Muestra `g1.txt` válido.
2. Corre el programa: se ven los **4 pasos** y la **CNF** resultante.
3. Edita `g1.txt` para introducir un error (ej. `s -> a` con minúscula en LHS), vuelve a correr y muestra el **rechazo por regex**.
4. Repite con `g2.txt` (breve).
5. Sube el video **no listado** y agrega el enlace al `README.md`. :contentReference[oaicite:5]{index=5}
