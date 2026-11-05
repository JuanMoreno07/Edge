# -*- coding: utf-8 -*-
"""
Mini-caso tecnico: scheduling de inferencia en el borde de la RAN.
Compara una politica "liviana" aprendida (regresion logistica con numpy)
contra heuristicas FCFS y SJF.

KPIs: throughput, latencia de inferencia (promedio y p99), tasa de perdida de
plazo (deadline miss), utilizacion de CPU y proxy de consumo energetico.
"""
import os
import numpy as np
import heapq

rng = np.random.default_rng(20261105)  # semilla reproducible

# ---------- Parametros del escenario ----------
C = 10.0            # tasa de computo del servidor de borde (GFLOP/s)
LAMBDA = 3.0        # tasa de llegadas de tareas (tareas/s)
SIM_TIME = 120.0    # horizonte de simulacion (s)
P_BUSY = 100.0      # potencia con CPU ocupada (W)
P_IDLE = 20.0       # potencia en reposo (W)
P_URLLC = 0.30      # fraccion de tareas URLLC

# ---------- Generacion de tareas ----------
def generate_tasks():
    tasks = []
    t = 0.0
    seq = 0
    while t < SIM_TIME:
        inter = rng.exponential(1.0 / LAMBDA)
        t += inter
        if t >= SIM_TIME:
            break
        is_urllc = rng.random() < P_URLLC
        demand = rng.uniform(0.5, 5.0)                 # GFLOP
        if is_urllc:
            slack = rng.uniform(0.1, 1.0)              # s (estricto)
        else:
            slack = rng.uniform(1.0, 5.0)              # s (holgado)
        service = demand / C
        deadline = t + slack + service                 # plazo absoluto
        tasks.append(dict(arrival=t, demand=demand, slack=slack,
                          urllc=int(is_urllc), service=service,
                          deadline=deadline, seq=seq))
        seq += 1
    return tasks

# ---------- Simulador no apropiativo ----------
def simulate(tasks, policy, model=None, feats=None):
    n = len(tasks)
    finish = [0.0]*n
    missed = [0]*n
    ready = []           # heap de (clave, seq, idx)
    i = 0
    t = 0.0
    busy_until = 0.0
    total_busy = 0.0
    while i < n or ready:
        if not ready:
            t = max(t, tasks[i]['arrival'])
        # encolar todas las tareas llegadas hasta t
        while i < n and tasks[i]['arrival'] <= t + 1e-12:
            tk = tasks[i]
            if policy == 'fcfs':
                key = tk['arrival']
            elif policy == 'sjf':
                key = tk['demand']
            elif policy == 'learned':
                x = feats[i]
                Xn = (x - model['mu']) / model['sd']
                z = Xn.dot(model['w']) + model['b']
                p_miss = 1.0 / (1.0 + np.exp(-z))
                key = -float(p_miss)   # mayor riesgo -> menor clave -> primero
            else:
                key = 0.0
            heapq.heappush(ready, (key, tk['seq'], i))
            i += 1
        key, s, idx = heapq.heappop(ready)
        tk = tasks[idx]
        st = max(t, tk['arrival'], busy_until)
        finish[idx] = st + tk['service']
        busy_until = finish[idx]
        total_busy += tk['service']
        missed[idx] = 1 if finish[idx] > tk['deadline'] else 0
        t = finish[idx]
    total_time = max(SIM_TIME, busy_until)
    lat = [finish[k] - tasks[k]['arrival'] for k in range(n)]
    latency = np.array(lat)
    throughput = n / total_time
    util = total_busy / total_time
    energy = (P_BUSY * util + P_IDLE * (1 - util)) * total_time
    energy_per_task = energy / n
    n_ok = n - int(sum(missed))
    # desglose de fallas por clase
    miss_urllc = sum(1 for k in range(n) if missed[k] and tasks[k]['urllc'])
    miss_embb = sum(1 for k in range(n) if missed[k] and not tasks[k]['urllc'])
    n_urllc = sum(1 for k in range(n) if tasks[k]['urllc'])
    n_embb = n - n_urllc
    return dict(
        n=n, throughput=throughput, util=util,
        avg_latency=float(latency.mean()),
        p99_latency=float(np.percentile(latency, 99)),
        miss_rate=float(np.mean(missed)),
        energy_Ws=energy, energy_per_task=energy_per_task,
        energy_per_useful=energy / n_ok if n_ok else float('inf'),
        missed=int(sum(missed)), missed_list=missed,
        miss_urllc=miss_urllc, miss_embb=miss_embb,
        n_urllc=n_urllc, n_embb=n_embb,
        total_time=total_time,
    )

# ---------- Caracteristicas en llegada (tareas en sistema) ----------
def arrival_features(tasks):
    n = len(tasks)
    n_in = [0]*n
    busy_until = 0.0
    finish = [0.0]*n
    arrivals = sorted(range(n), key=lambda k: tasks[k]['arrival'])
    for k in arrivals:
        tk = tasks[k]
        n_in[k] = sum(1 for j in range(n)
                      if j != k and tasks[j]['arrival'] <= tk['arrival'] < finish[j])
        st = max(tk['arrival'], busy_until)
        finish[k] = st + tk['service']
        busy_until = finish[k]
    return n_in

# ---------- Modelo liviano: regresion logistica ----------
def train_logreg(X, y, lr=0.1, epochs=800):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = Xn.dot(w) + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (Xn.T.dot(g) / len(y))
        b -= lr * g.mean()
    return dict(w=w, b=b, mu=mu, sd=sd)

if __name__ == "__main__":
    tasks = generate_tasks()
    n = len(tasks)
    print(f"Tareas generadas: {n}")
    mean_demand = np.mean([t['demand'] for t in tasks])
    load = LAMBDA * mean_demand / C
    print(f"Carga ofrecida (rho) = lambda*E[demanda]/C = {LAMBDA}*{mean_demand:.2f}/{C} = {load:.3f}")
    n_in = arrival_features(tasks)

    # Etiquetas de entrenamiento: ¿falla el plazo bajo FCFS?
    base = simulate(tasks, 'fcfs')
    y = np.array(base['missed_list'], dtype=float)

    X = np.array([[tasks[k]['demand'], tasks[k]['slack'], tasks[k]['urllc'], n_in[k]]
                  for k in range(n)])
    model = train_logreg(X, y)
    print("Pesos del modelo (demand, slack, urllc, n_in):",
          np.round(model['w'], 3), "bias=", round(float(model['b']), 3))
    print("Tasa base de falla (FCFS) usada como etiquetas: "
          f"{y.mean()*100:.1f}%")

    res = {}
    for name in ['fcfs', 'sjf', 'learned']:
        res[name] = simulate(tasks, name, model=model, feats=X)

    print("\n=== RESULTADOS ===")
    print(f"{'Politica':<10} {'Thr(t/s)':>9} {'Lat avg(s)':>10} {'Lat p99(s)':>10} "
          f"{'Miss%':>7} {'CPU%':>6} {'E/task(J)':>9}")
    for name in ['fcfs', 'sjf', 'learned']:
        r = res[name]
        print(f"{name:<10} {r['throughput']:>9.3f} {r['avg_latency']:>10.3f} "
              f"{r['p99_latency']:>10.3f} {r['miss_rate']*100:>6.1f}% "
              f"{r['util']*100:>5.1f}% {r['energy_per_task']:>9.2f}")
    print(f"\nTareas totales: {n} | fallas de plazo -> "
          f"FCFS={res['fcfs']['missed']}, SJF={res['sjf']['missed']}, "
          f"learned={res['learned']['missed']}")
    print(f"\nDesglose por clase (URLLC={res['fcfs']['n_urllc']} tareas, "
          f"eMBB={res['fcfs']['n_embb']} tareas):")
    print(f"{'Politica':<10} {'Miss URLLC':>11} {'Miss eMBB':>10}")
    for name in ['fcfs', 'sjf', 'learned']:
        r = res[name]
        print(f"{name:<10} {r['miss_urllc']:>9}  {r['miss_embb']:>9}")

    # CSV para el blog
    with open(os.path.join(os.path.dirname(__file__), "mini_caso_resultados.csv"),
              "w", encoding="utf-8") as f:
        f.write("Politica,Throughput(t/s),Latencia_prom(s),Latencia_p99(s),"
                "Tasa_falla(%),Utilizacion_CPU(%),Energia_por_tarea(J),"
                "Energia_por_tarea_util(J),Fallas_URLLC,Fallas_eMBB\n")
        for name in ['fcfs', 'sjf', 'learned']:
            r = res[name]
            f.write(f"{name},{r['throughput']:.4f},{r['avg_latency']:.4f},"
                    f"{r['p99_latency']:.4f},{r['miss_rate']*100:.2f},"
                    f"{r['util']*100:.2f},{r['energy_per_task']:.3f},"
                    f"{r['energy_per_useful']:.3f},{r['miss_urllc']},"
                    f"{r['miss_embb']}\n")
    print("\nCSV guardado: mini_caso_resultados.csv")
