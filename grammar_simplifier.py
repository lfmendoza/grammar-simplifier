#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grammar Simplifier
- Carga gramáticas desde archivos de texto (una producción por línea; OR con '|').
- Valida cada línea con una regex estricta (no-terminales en MAYÚSCULA, terminales en minúscula o dígitos, ε como 'ε' o 'eps').
- Elimina producciones-ε (epsilon), unitarias, símbolos inútiles (no generadores y no alcanzables).
- Convierte a Forma Normal de Chomsky (CNF).
- Imprime paso a paso cada transformación.

Formato de entrada:
S -> 0A0 | 1B1 | BB
A -> C
B -> S | A
C -> S | ε

Convenciones:
- No-terminal: una sola letra A..Z
- Terminal: una sola letra a..z o dígito 0..9
- Epsilon: 'ε' o 'eps'
- Flecha: '->'
- OR: '|'
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Iterable
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict, deque
from itertools import product

EPS = "ε"

# Regex para línea de producción: LHS -> RHS1 | RHS2 | ...
# - LHS: [A-Z]
# - RHS: secuencia de símbolos [A-Z]|[a-z0-9]|ε (sin espacios internos), o vacío si es ε explícita
# - espacios permitidos alrededor de '->' y '|'
LHS_RE = r"[A-Z]"
SYM_RE = r"(?:[A-Z]|[a-z0-9])"
EPS_RE = r"(?:ε|eps)"
RHS_PIECE_RE = rf"(?:{SYM_RE}+|{EPS_RE})"
LINE_RE = re.compile(
    rf"^\s*({LHS_RE})\s*->\s*({RHS_PIECE_RE})(?:\s*\|\s*({RHS_PIECE_RE}))*\s*$"
)

# Para tokenizar RHS con símbolos unitarios (1 char)
TOKEN_RE = re.compile(r"[A-Z]|[a-z0-9]")

@dataclass
class Grammar:
    start: str
    prods: Dict[str, Set[Tuple[str, ...]]] = field(default_factory=lambda: defaultdict(set))

    @property
    def nonterminals(self) -> Set[str]:
        return set(self.prods.keys())

    @property
    def terminals(self) -> Set[str]:
        ts: Set[str] = set()
        for rhss in self.prods.values():
            for rhs in rhss:
                for s in rhs:
                    if s not in self.prods and s != EPS and not s.isupper():
                        ts.add(s)
                    if s.isdigit():
                        ts.add(s)
        return ts

    def add_prod(self, lhs: str, rhs: Tuple[str, ...]) -> None:
        self.prods[lhs].add(rhs)

    def clone(self) -> "Grammar":
        g = Grammar(self.start)
        for A, rhss in self.prods.items():
            g.prods[A] = set(rhss)
        return g

    def __str__(self) -> str:
        lines = []
        nts = sorted(self.prods.keys())
        for A in nts:
            rhss = sorted(["".join(rhs) if rhs != (EPS,) else EPS for rhs in self.prods[A]])
            lines.append(f"{A} -> " + " | ".join(rhss))
        return "\n".join(lines)

def parse_grammar(text: str) -> Grammar:
    """
    Parsea texto con producciones. La primera LHS encontrada se usa como símbolo inicial.
    Valida cada línea con regex. Si una línea no es válida, aborta con ValueError.
    """
    g = None
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if not LINE_RE.match(line):
            raise ValueError(f"Línea {i} inválida: {line!r}")
        lhs, rest = line.split("->", 1)
        lhs = lhs.strip()
        if g is None:
            g = Grammar(start=lhs)
        rhs_parts = [p.strip() for p in rest.split("|")]
        for part in rhs_parts:
            if part in ("ε", "eps"):
                rhs = (EPS,)
            else:
                toks = TOKEN_RE.findall(part)
                if not toks or "".join(toks) != part:
                    raise ValueError(f"Línea {i}: RHS inválido {part!r}")
                rhs = tuple(toks)
            g.add_prod(lhs, rhs)
    if g is None:
        raise ValueError("No se encontraron producciones.")
    return g

def from_file(path: Path) -> Grammar:
    txt = path.read_text(encoding="utf-8")
    return parse_grammar(txt)

# ---------- Utilidades de impresión ----------

def print_step(title: str, g: Grammar) -> None:
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(g)
    print()

# ---------- 1) Eliminación de ε-producciones ----------

def nullable_symbols(g: Grammar) -> Set[str]:
    /*
    Un no-terminal A es anulable si A =>* ε.
    Regla:
      - Si A -> ε, entonces A es anulable.
      - Si A -> X1 ... Xk y todos Xi son anulables, entonces A es anulable.
    */
    nulls: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for A, rhss in g.prods.items():
            if A in nulls:
                continue
            for rhs in rhss:
                if rhs == (EPS,):
                    nulls.add(A)
                    changed = True
                    break
                if all((X in nulls) for X in rhs if X.isupper()):
                    # todos no-terminales anulables y no hay terminales
                    if all(X.isupper() for X in rhs) or len(rhs) == 0:
                        nulls.add(A)
                        changed = True
                        break
    return nulls

def remove_epsilon(g: Grammar, verbose=True) -> Grammar:
    """
    Elimina A -> ε salvo que Start =>* ε (en cuyo caso se conserva una forma controlada).
    Para cada A -> X1 ... Xk con m símbolos anulables en {Xi}, genera 2^m combinaciones
    (excluyendo la que elimina todos si corresponde a ε pura, salvo caso de Start).
    """
    if verbose:
        print("Paso 1: Eliminación de producciones-ε")
    g2 = g.clone()
    nulls = nullable_symbols(g2)
    if verbose:
        print(f"Símbolos anulables: {sorted(nulls)}")

    new_prods: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    for A, rhss in g2.prods.items():
        for rhs in rhss:
            if rhs == (EPS,):
                # diferimos decisión: se eliminará después si procede
                continue
            # indices de posiciones anulables (solo no-terminales anulables)
            idxs = [i for i, X in enumerate(rhs) if X.isupper() and X in nulls]
            # combinaciones binarias de eliminar/retener cada anulable
            for mask in product([0,1], repeat=len(idxs)):
                out = list(rhs)
                # eliminar los Xi donde mask[j] == 1 (desde el final para no mover índices)
                for j, bit in sorted(list(enumerate(mask)), key=lambda t: -t[0]):
                    if bit == 1:
                        del out[idxs[j]]
                if len(out) == 0:
                    # ε candidato
                    new_prods[A].add((EPS,))
                else:
                    new_prods[A].add(tuple(out))
        # si alguna producción era explícitamente ε, la combinatoria ya lo cubre
    # Manejo del Start si no era anulable:
    # Regla usual: si S es anulable originalmente, permitimos S -> ε; si no, removemos ε.
    for A, rhss in new_prods.items():
        pass
    # Construir g3 sin ε, salvo si start anulable
    g3 = Grammar(g2.start)
    start_nullable = g2.start in nulls
    for A, rhss in new_prods.items():
        for rhs in rhss:
            if rhs == (EPS,) and not (start_nullable and A == g2.start):
                # descartar ε si no es el Start anulable
                continue
            g3.add_prod(A, rhs)

    if verbose:
        print_step("Gramática tras eliminar ε-producciones", g3)
    return g3

# ---------- 2) Eliminación de producciones unitarias ----------

def remove_unit(g: Grammar, verbose=True) -> Grammar:
    """
    Quita A -> B (ambos no-terminales). Cierra transitivamente y copia producciones no unitarias.
    """
    if verbose:
        print("Paso 2: Eliminación de producciones unitarias")
    g2 = g.clone()
    # cierre unitario: para cada A, conjunto de B tales que A =>* B por reglas unitarias
    unit_closure: Dict[str, Set[str]] = {A: {A} for A in g2.nonterminals}
    for A in g2.nonterminals:
        changed = True
        while changed:
            changed = False
            for B in list(unit_closure[A]):
                for rhs in g2.prods[B]:
                    if len(rhs) == 1 and rhs[0].isupper():
                        C = rhs[0]
                        if C not in unit_closure[A]:
                            unit_closure[A].add(C)
                            changed = True

    g3 = Grammar(g2.start)
    for A in g2.nonterminals:
        for B in unit_closure[A]:
            for rhs in g2.prods[B]:
                # omitir reglas unitarias puras
                if len(rhs) == 1 and rhs[0].isupper():
                    continue
                g3.add_prod(A, rhs)

    if verbose:
        print_step("Gramática tras eliminar unitarias", g3)
    return g3

# ---------- 3) Eliminación de símbolos inútiles ----------

def generating_nonterminals(g: Grammar) -> Set[str]:
    """
    No-terminal generador si A =>* w con w en Σ*.
    """
    gen: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for A, rhss in g.prods.items():
            if A in gen:
                continue
            for rhs in rhss:
                # todos los símbolos deben ser terminales o no-terminales ya generadores
                ok = True
                for s in rhs:
                    if s.isupper() and s not in gen:
                        ok = False
                        break
                    if s == EPS:
                        # ε cuenta como cadena terminal
                        ok = True
                if ok:
                    gen.add(A)
                    changed = True
                    break
    return gen

def reachable_symbols(g: Grammar) -> Set[str]:
    """
    Símbolos alcanzables desde S.
    """
    reach: Set[str] = {g.start}
    q = deque([g.start])
    while q:
        A = q.popleft()
        for rhs in g.prods.get(A, ()):
            for s in rhs:
                if s.isupper() and s not in reach:
                    reach.add(s)
                    q.append(s)
    return reach

def remove_useless(g: Grammar, verbose=True) -> Grammar:
    if verbose:
        print("Paso 3: Eliminación de símbolos inútiles")

    # 3a: remover no-generadores
    gen = generating_nonterminals(g)
    g1 = Grammar(g.start)
    for A in g.nonterminals & gen:
        for rhs in g.prods[A]:
            if all((not s.isupper()) or (s in gen) for s in rhs):
                g1.add_prod(A, rhs)
    if verbose:
        print_step(f"Tras remover no-generadores (generadores={sorted(gen)})", g1)

    # 3b: remover no-alcanzables
    reach = reachable_symbols(g1)
    g2 = Grammar(g.start)
    for A in g1.nonterminals & reach:
        for rhs in g1.prods[A]:
            g2.add_prod(A, rhs)
    if verbose:
        print_step(f"Tras remover no-alcanzables (alcanzables={sorted(reach)})", g2)
    return g2

# ---------- 4) Conversión a CNF ----------

def to_cnf(g: Grammar, verbose=True) -> Grammar:
    """
    Convierte a CNF (A->BC o A->a), preservando S->ε si la gramática original lo requiere.
    Pasos:
      - Asegurar que no haya ε ni unitarias ni inútiles (se asume que ya pasamos remove_epsilon/unit/useless).
      - Reemplazar terminales en reglas largas por nuevos no-terminales (T_a -> a).
      - Binarizar reglas con longitud >= 3: A->X1...Xk -> A->X1 Z1, Z1->X2 Z2, ..., Z_{k-3}->X_{k-2} X_{k-1}X_k (binarizado).
    """
    if verbose:
        print("Paso 4: Conversión a CNF")
    g2 = g.clone()

    # 4.1: Introducir no-terminales para terminales en RHS de longitud > 1
    term_map: Dict[str, str] = {}
    counter = 1

    def fresh_T(a: str) -> str:
        nonlocal counter
        if a not in term_map:
            term_map[a] = f"T_{a}"
        return term_map[a]

    cnf = Grammar(g2.start)
    # Agregar T_a -> a según se requiera
    for A, rhss in g2.prods.items():
        for rhs in rhss:
            if len(rhs) == 1 and not rhs[0].isupper() and rhs[0] != EPS:
                # A -> a (ya cumple CNF)
                cnf.add_prod(A, rhs)
            else:
                # reemplazar terminales en RHS largos
                new_rhs: List[str] = []
                for s in rhs:
                    if not s.isupper() and s != EPS:
                        T = fresh_T(s)
                        new_rhs.append(T)
                    else:
                        new_rhs.append(s)
                cnf.add_prod(A, tuple(new_rhs))

    # Declarar T_a -> a
    for a, T in term_map.items():
        cnf.add_prod(T, (a,))

    # 4.2: Binarizar A -> X1 X2 ... Xk, k>=3
    def binarize(A: str, rhs: Tuple[str, ...]) -> List[Tuple[str, Tuple[str, ...]]]:
        """
        Retorna lista de (LHS, RHS) tras binarizar.
        """
        if len(rhs) <= 2:
            return [(A, rhs)]
        created = []
        symbols = list(rhs)
        left = symbols[0]
        rest = symbols[1:]
        prev = left
        while len(rest) > 1:
            X, *rest = rest
            new_nt = fresh_intermediate(A, prev, X)
            created.append((A if prev == left else prev_nt, (prev, new_nt)))
            prev_nt = new_nt
            prev = X
            A = prev_nt
        # última pareja
        created.append((A if prev == left else prev_nt, (prev, rest[0])))
        return created

    inter_counter = defaultdict(int)

    def fresh_intermediate(*args) -> str:
        key = "_".join(args)
        inter_counter[key] += 1
        idx = inter_counter[key]
        base = "X"
        return f"{base}{idx}"

    # Binarización estable:
    changed = True
    while changed:
        changed = False
        new_prods = defaultdict(set)
        for A, rhss in cnf.prods.items():
            for rhs in rhss:
                if len(rhs) <= 2:
                    new_prods[A].add(rhs)
                else:
                    changed = True
                    # estrategia simple: encadenar derecha
                    prev_nt = A
                    seq = list(rhs)
                    while len(seq) > 2:
                        X1 = seq.pop(0)
                        X2 = seq.pop(0)
                        new_nt = fresh_intermediate(prev_nt, X1, X2)
                        new_prods[prev_nt].add((X1, new_nt))
                        prev_nt = new_nt
                        seq.insert(0, X2)
                    new_prods[prev_nt].add(tuple(seq))
        cnf.prods = new_prods

    if verbose:
        print_step("Gramática en CNF", cnf)
    return cnf

# ---------- Pipeline completo ----------

def simplify_to_cnf(g: Grammar, verbose=True) -> Tuple[Grammar, Grammar, Grammar, Grammar]:
    g1 = remove_epsilon(g, verbose=verbose)
    g2 = remove_unit(g1, verbose=verbose)
    g3 = remove_useless(g2, verbose=verbose)
    g4 = to_cnf(g3, verbose=verbose)
    return g1, g2, g3, g4

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Grammar Simplifier (ε, unit, useless, CNF)")
    ap.add_argument("file", help="Ruta del archivo de gramática (txt)")
    ap.add_argument("--no-steps", action="store_true", help="No imprimir pasos intermedios")
    args = ap.parse_args()

    g = from_file(Path(args.file))
    if not args.no_steps:
        print_step("Gramática original (validada por regex)", g)
    _, _, _, _ = simplify_to_cnf(g, verbose=not args.no_steps)

if __name__ == "__main__":
    main()
