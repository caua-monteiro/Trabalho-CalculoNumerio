import math as mt
import sympy as sp

def Bissecao(a: float, b: float, delta: float, n: int, exp_str: str):
    iter = []
    count = 0
    if abs(a-b) < delta:
        return a, count
    while(abs(a-b) > delta and count < 50):
        count += 1
        meio = (a+b)/2.0
        fInicio = funcao(a, exp_str)
        fMeio = funcao(meio, exp_str)

        iter.append((count, a, b, meio, fMeio))
        if fInicio*fMeio < 0:
            b = meio
        else:
            a = meio

    return meio, iter

def MIL(a: float, delta: float, n: int, exp_str: str, equac_equiv: str):
    iter = []
    
    count = 0

    while (count < n):
        f = funcao(a, exp_str)
        var = funcao(a, equac_equiv)

        iter.append((count, a, var, f))

        if(abs(f) < delta or var-a < delta):
            break
        a = var
        count += 1
    return a, iter


def Newton(a: float, delta: float, n: int, exp_str: str):
    iter = []

    count = 1
    x = sp.Symbol('x')
    exp = sp.sympify(exp_str)
    exp_deriv = sp.diff(exp, x)

    f = sp.lambdify(x, exp, 'math')
    f_linha = sp.lambdify(x, exp_deriv, 'math')

    count = 0

    while count < n:
        fa = f(a)
        fpa = f_linha(a)
        if fpa == 0:
            break
        novo = a - fa / fpa
        iter.append((count, a, novo, fa))
        if abs(novo - a) < delta or abs(fa) < delta:
            break
        a = novo
        count += 1
    return a, iter

def Secante(a: float, b: float, delta: float, n: int, exp_str: str):
    iteracoes = []
    f = lambda x: funcao(x, exp_str)
    count = 0
    while count < n:
        fa, fb = f(a), f(b)
        if fb - fa == 0:
            break
        var = b - (fb * (b - a)) / (fb - fa)
        iteracoes.append((count, a, b, var, f(var)))
        if abs(var - b) < delta or abs(f(var)) < delta:
            break
        a, b = b, var
        count += 1
    return a, iteracoes


def Regula_falsi(a: float, b: float, delta: float, n: int, exp_str: str):
    iteracoes = []
    f = lambda x: funcao(x, exp_str)
    fa, fb = f(a), f(b)
    count = 0
    while count < n:
        try:
            var = (a * fb - b * fa) / (fb - fa)
            fvar = f(var)
            iteracoes.append((count, a, b, var, fvar))
            if abs(fvar) < delta or abs(b - a) < delta:
                break
            if fa * fvar < 0:
                b = var
                fb = fvar
            else:
                a = var
                fa = fvar
            count += 1
        except ZeroDivisionError:
            print("Divisao por 0 encontrada")
        
    return var, iteracoes


def funcao(varivel: float, exp_str: str):
    x = sp.Symbol('x')
    funcao_str = 0
    exp_real = sp.sympify(funcao_str)
    funcao_real = sp.lambdify(x, exp_real, 'math')

    return funcao_real(varivel)


def lerArquivos(nomeArquivo: str):
    try:
        with open(nomeArquivo, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()
            return conteudo
    except FileNotFoundError:
        print("ARQUIVO NAO ENCONTRADO")
        return ""

def salvar_resultados(nome_saida: str, resultados: dict):
    with open(nome_saida, 'w', encoding='utf-8') as f:
        for metodo, dados in resultados.items():
            f.write(f"===== {metodo} =====\n")
            f.write("Iter\tValores...\n")
            for linha in dados[1]:
                f.write("\t".join(map(str, linha)) + "\n")
            f.write(f"\nRaiz aproximada: {dados[0]}\n\n\n")
    print(f"Resultados salvos em {nome_saida}")


def main():
    entrada = lerArquivos('entrada.txt')
    if not entrada:
        return
    a, b, delta, n, funcao, equac_equiv = entrada.split(";")
    a = float(a)
    b = float(b)
    delta = float(delta)
    n = int(n)

    resultados = {}

    resultados["Bissecao"] = Bissecao(a, b, delta, n, funcao)
    resultados["MIL (Ponto Fixo)"] = MIL(a, delta, n, funcao, equac_equiv)
    resultados["Newton-Raphson"] = Newton(a, delta, n, funcao)
    resultados["Secante"] = Secante(a, b, delta, n, funcao)
    resultados["Regula Falsi"] = Regula_falsi(a, b, delta, n, funcao)

    salvar_resultados("saida.txt", resultados)

if __name__ == "__main__":
    main()