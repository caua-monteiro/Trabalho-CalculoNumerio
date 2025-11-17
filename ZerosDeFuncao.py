import math as mt
import sympy as sp
from typing import Tuple, List, Any

class IteracoesInsuficientesError(Exception):
    "O numero de iteracoes nao foi suficiente para convergir"
    pass

x = sp.Symbol('x')

def makeCallable(strExpr: str):
    if strExpr is None or strExpr.strip() == "":
        raise ValueError("Expressão vazia ou nula")
    
    strExpr = strExpr.replace('^', '**')

    try:
        expr = sp.sympify(strExpr, evaluate=True)
    except sp.SympifyError as e:
        raise ValueError(f"Erro ao interpretar a expressão{strExpr}: {e}")
    
    func = sp.lambdify(x, expr, modules=['math'])
    return func, expr

def Bissecao(a: float, b: float, delta: float, n: int, exp_str: str) -> Tuple[float, List[Tuple[Any, ...]]]:
    f, _ = makeCallable(exp_str)
    iter_list = []
    count = 0
    if abs(a-b) < delta:
        return a, count
    
    fa, fb = f(a), f(b)
    while abs(a-b) > delta:
        count += 1

        if count > n:
            raise IteracoesInsuficientesError(f"A funcao {exp_str} nao pode convergir em {n} iteracoes")
        
        meio = (a+b)/2.0
        
        fMeio = f(meio)

        iter_list.append((count, a, b, meio, fMeio))
        if fa * fMeio < 0:
            b = meio
            fb = fMeio
        else:
            a = meio
            fa = fMeio

    return meio, iter_list

def MIL(a: float, delta: float, n: int, exp_str: str, equac_equiv: str) -> Tuple[float, List[Tuple[any, ...]]]:
    f, _ = makeCallable(exp_str)
    g, _ = makeCallable(equac_equiv)
    
    iter_list = []
    
    kx = a

    count = 0

    while (count < n):
        gx = g(kx)
        fx = f(kx)

        iter_list.append((count, kx, gx, fx))

        if(abs(fx) < delta or abs(gx - kx) < delta):
            kx = gx
            break
        
        kx = gx
        count += 1
    
    if count == n:
        raise IteracoesInsuficientesError(f"A funcao {exp_str} nao pode convergir em {n} iteracoes")

    return kx, iter_list


def Newton(a: float, delta: float, n: int, exp_str: str) -> Tuple[float, List[Tuple[Any, ...]]]:
    iter_list = []

    try:
        expr = sp.sympify(exp_str.replace('^', '**'), evaluate=True)
    except sp.SympifyError as e:
        raise ValueError(f"Erro ao interpretar a expressão {exp_str}: {e}")
    
    expr_deriv = sp.diff(expr, x)
    f = sp.lambdify(x, expr, modules=['math'])
    fp = sp.lambdify(x, expr_deriv, modules=['math'])
    xk = a
    count = 0

    while count < n:
        fx = f(xk)
        fpx = fp(xk)
        if fpx == 0:
            iter_list.append((count, xk, None, fx, "Derivada zero"))
            break
        Xn = xk - fx / fpx
        iter_list.append((count + 1, xk, Xn, fx, fpx))
        if abs(Xn - xk) < delta or abs(fx) < delta:
            xk = Xn
            break
        xk = Xn
        count += 1
    
    if count == n:
        raise IteracoesInsuficientesError(f"A funcao {exp_str} nao pode convergir em {n} iteracoes")

    return xk, iter_list

def Secante(a: float, b: float, delta: float, n: int, exp_str: str) -> Tuple[float, List[Tuple[Any, ...]]]:
    f, _ = makeCallable(exp_str)
    iter_list = []
    x0, x1 = a, b
    count = 0

    while count < n:
        f0, f1 = f(x0), f(x1)
        if f0 - f1 == 0:
            iter_list.append((count, x0, x1, None, "Divisao por zero"))
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        iter_list.append((count, x0, x1, x2, f(x2)))
        if abs(x2 - x1) < delta or abs(f(x2)) < delta:
            x1 = x2
            break
        x0, x1 = x1, x2
        count += 1

    if count == n:
        raise IteracoesInsuficientesError(f"A funcao {exp_str} nao pode convergir em {n} iteracoes")
    
    return x1, iter_list


def Regula_falsi(a: float, b: float, delta: float, n: int, exp_str: str) -> Tuple[float, List[Tuple[Any, ...]]]:
    f, _ = makeCallable(exp_str)
    iter_list = []
    
    fa, fb = f(a), f(b)
    count = 0
    var = None
    while count < n:
        if fa - fb == 0:
            iter_list.append((count, a, b, None, "Divisao por zero"))
            break

        var = (a * fb - b * fa) / (fb - fa)
        fvar = f(var)
        iter_list.append((count + 1, a, b, var, fvar))
        if abs(fvar) < delta or abs(b - a) < delta:
            break
        if fa * fvar < 0:
            b = var
            fb = fvar
        else:
            a = var
            fa = fvar
        count += 1
    
    if count == n:
        raise IteracoesInsuficientesError(f"A funcao {exp_str} nao pode convergir em {n} iteracoes")
        
    return var, iter_list

def lerArquivos(nomeArquivo: str) -> Tuple[float, float, float, int, str, str]:
    try:
        with open(nomeArquivo, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read().strip()
            if not conteudo:
                raise ValueError("Arquivo de entrada vazio.")
            # Espera: a;b;delta;n;funcao;equac_equiv
            partes = [p.strip() for p in conteudo.split(";")]
            if len(partes) < 5:
                raise ValueError("Arquivo de entrada deve conter pelo menos 5 campos separados por ';': a;b;delta;n;funcao;[equac_equiv opcional]")
            a = float(partes[0])
            b = float(partes[1])
            delta = float(partes[2])
            n = int(partes[3])
            func_expr = partes[4]
            equac_equiv = partes[5] if len(partes) > 5 else ""
            return a, b, delta, n, func_expr, equac_equiv
    except FileNotFoundError:
        raise FileNotFoundError("ARQUIVO NAO ENCONTRADO: " + nomeArquivo)
    except FileNotFoundError:
        print("ARQUIVO NAO ENCONTRADO")
        return ""

def salvar_resultados(nome_saida: str, resultados: dict):
    with open(nome_saida, 'w', encoding='utf-8') as f:
        for metodo, dados in resultados.items():
            raiz, iteracoes = dados
            f.write(f"===== {metodo} =====\n")
            if not iteracoes:
                f.write("Sem iteracoes ou metodo nao executado.\n")
            else:
                f.write("Iter\tDados...\n")
                for linha in iteracoes:
                    linha_str = "\t".join(str(item) for item in linha)
                    f.write(linha_str + "\n")
            f.write(f"Raiz aproximada: {raiz}\n\n")
    print(f"Resultados salvos em {nome_saida}")


def main():
    try:
        a, b, delta, n, func_expr, equac_equiv = lerArquivos('entrada.txt')
    except Exception as e:
        print("Erro na leitura do arquivo de entrada:", e)
        return

    resultados = {}
    try:
        resultados["Bissecao"] = Bissecao(a, b, delta, n, func_expr)
    except Exception as e:
        resultados["Bissecao"] = (None, [("erro", str(e))])

    try:
        if equac_equiv:
            resultados["MIL (Ponto Fixo)"] = MIL(a, delta, n, func_expr, equac_equiv)
        else:
            resultados["MIL (Ponto Fixo)"] = (None, [("info", "sem equac_equiv fornecida")])
    except Exception as e:
        resultados["MIL (Ponto Fixo)"] = (None, [("erro", str(e))])

    try:
        resultados["Newton-Raphson"] = Newton(a, delta, n, func_expr)
    except Exception as e:
        resultados["Newton-Raphson"] = (None, [("erro", str(e))])

    try:
        resultados["Secante"] = Secante(a, b, delta, n, func_expr)
    except Exception as e:
        resultados["Secante"] = (None, [("erro", str(e))])

    try:
        resultados["Regula Falsi"] = Regula_falsi(a, b, delta, n, func_expr)
    except Exception as e:
        resultados["Regula Falsi"] = (None, [("erro", str(e))])

    salvar_resultados("saida.txt", resultados)

if __name__ == "__main__":
    main()