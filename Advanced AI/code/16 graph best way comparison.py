import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# =====================================================
# 1) ROAD NETWORK
# =====================================================

class RoadNetwork:

    def __init__(self):
        self.G = nx.Graph()

        self.pos = {
            0:(0,4), 1:(1,5), 2:(1,3),
            3:(3,6), 4:(3,4), 5:(3,2),
            6:(5,5), 7:(5,3), 8:(5,1),
            9:(7,4), 10:(7,2), 11:(7,0),
            12:(9,3), 13:(9,1), 14:(11,2)
        }

        self.build()

    # -------------------------------------------------
    # Reliability model (independent design)
    # -------------------------------------------------
    def reliability_model(self, length, speed, fuel):

        base = 0.90

        speed_penalty = min(speed / 100, 0.05)
        fuel_penalty  = min(fuel / 100, 0.05)
        length_bonus  = min(length / 100, 0.15)

        r = base + length_bonus - speed_penalty - fuel_penalty

        return max(0.6, min(0.99, r))

    # -------------------------------------------------
    def build(self):

        edges = [
            (0,1,10,60,8,1.0),
            (0,2,25,120,12,1.0),

            (1,3,8,40,6,3.5),
            (1,4,15,80,10,1.2),

            (2,4,7,110,8,1.0),
            (2,5,12,130,9,1.0),

            (3,6,6,30,4,4.0),
            (4,6,10,90,7,1.5),

            (4,7,15,120,10,1.0),
            (5,7,8,140,6,1.0),

            (5,8,12,150,8,1.0),
            (6,9,7,40,5,4.0),

            (7,9,20,130,12,1.0),
            (7,10,10,120,7,1.0),

            (8,10,6,140,4,1.0),
            (8,11,7,160,5,1.0),

            (9,12,6,40,4,4.0),
            (10,12,9,120,7,1.0),

            (10,13,8,120,6,1.0),
            (11,13,4,170,3,1.0),

            (12,14,10,50,7,3.0),
            (13,14,12,150,8,1.0)
        ]

        for u,v,l,s,f,t in edges:

            time_cost = (l/s)*t
            r = self.reliability_model(l,s,f)

            self.G.add_edge(
                u,v,
                length=l,
                speed=s,
                fuel=f,
                time=time_cost,
                reliability=r
            )


# =====================================================
# 2) WEIGHT SETTING (MULTI OBJECTIVE)
# =====================================================

def set_weights(G, wd, wt, wf, wr):

    for u,v,d in G.edges(data=True):

        d["w"] = (
            wd * d["length"]/25 +
            wt * d["time"]/2 +
            wf * d["fuel"]/12 +
            wr * (-math.log(d["reliability"]))
        )


# =====================================================
# 3) PATH EVALUATION
# =====================================================

def evaluate(G, path):

    d=t=f=0
    r=1

    for i in range(len(path)-1):

        u,v = path[i], path[i+1]
        e = G[u][v]

        d += e["length"]
        t += e["time"]
        f += e["fuel"]
        r *= e["reliability"]

    return d,t,f,r


# =====================================================
# 4) CLASSICAL ALGORITHMS
# =====================================================

def dijkstra(G,s,t):
    p = nx.dijkstra_path(G,s,t,weight="w")
    c = nx.path_weight(G,p,"w")
    return p,c


def astar(G,pos,s,t):

    def h(a,b):
        x1,y1=pos[a]
        x2,y2=pos[b]
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    p = nx.astar_path(G,s,t,heuristic=h,weight="w")
    c = nx.path_weight(G,p,"w")
    return p,c


def bellman(G,s,t):
    p = nx.bellman_ford_path(G,s,t,weight="w")
    c = nx.path_weight(G,p,"w")
    return p,c


# =====================================================
# 5) GENETIC ALGORITHM
# =====================================================

class GA:

    def __init__(self,G):
        self.G=G

    def random_path(self,s,t):

        for _ in range(50):

            p=[s]
            cur=s
            vis={s}

            while cur!=t:

                n=[x for x in self.G.neighbors(cur) if x not in vis]
                if not n: break

                cur=random.choice(n)
                p.append(cur)
                vis.add(cur)

            if cur==t:
                return p

        return None

    def cost(self,p):
        return nx.path_weight(self.G,p,"w")

    def solve(self,s,t):

        best=None
        best_cost=1e9

        for _ in range(200):

            p=self.random_path(s,t)
            if not p: continue

            c=self.cost(p)

            if c<best_cost:
                best_cost=c
                best=p

        return best,best_cost


# =====================================================
# 6) ANT COLONY OPTIMIZATION
# =====================================================

class ACO:

    def __init__(self,G):
        self.G=G
        self.ph={e:1 for e in G.edges()}
        self.ph.update({(v,u):1 for u,v in G.edges()})

    def prob(self,u,v):

        tau=self.ph.get((u,v),1)
        eta=1/self.G[u][v]["length"]

        return tau * (eta**2)

    def solve(self,s,t):

        best=None
        best_cost=1e9

        for _ in range(100):

            p=[s]
            cur=s
            vis={s}

            while cur!=t:

                n=[x for x in self.G.neighbors(cur) if x not in vis]
                if not n: break

                w=[self.prob(cur,x) for x in n]
                cur=random.choices(n,w)[0]

                p.append(cur)
                vis.add(cur)

            if cur==t:

                c=nx.path_weight(self.G,p,"w")

                if c<best_cost:
                    best_cost=c
                    best=p

        return best,best_cost


# =====================================================
# 7) VISUALIZATION
# =====================================================

def draw(G,pos,path,title,cost):

    plt.figure(figsize=(14,8))

    nx.draw(G,pos,with_labels=True,node_size=1200,node_color="lightblue")

    labels = {
        (u,v): f"L={d['length']}\nS={d['speed']}\nF={d['fuel']}\nR={d['reliability']:.2f}"
        for u,v,d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(G,pos,labels,font_size=7)

    edges=list(zip(path[:-1],path[1:]))

    nx.draw_networkx_edges(G,pos,edgelist=edges,width=4,edge_color="red")

    d,t,f,r = evaluate(G,path)

    plt.suptitle(title,fontsize=16,fontweight="bold")

    info = (
        f"Path: {path}\n"
        f"Distance: {d:.2f} km\n"
        f"Time: {t:.3f} hr\n"
        f"Fuel: {f:.2f}\n"
        f"Reliability: {r:.4f}\n"
        f"Objective: {cost:.5f}"
    )

    plt.text(0.02,0.98,info,transform=plt.gca().transAxes,
             bbox=dict(facecolor="white",alpha=0.9),
             va="top")

    plt.axis("off")
    plt.show()


# =====================================================
# 8) MAIN PROGRAM
# =====================================================

net = RoadNetwork()
G = net.G

start,goal = 0,14

ga = GA(G)
aco = ACO(G)

cases = [

("SHORTEST PATH",(1,0,0,0)),
("FASTEST PATH",(0,1,0,0)),
("MIN FUEL PATH",(0,0,1,0)),
("MOST RELIABLE PATH",(0,0,0,1)),
("MULTI OBJECTIVE",(0.35,0.35,0.2,0.1))
]

for name,w in cases:

    set_weights(G,*w)

    p1,c1 = dijkstra(G,start,goal)
    p2,c2 = astar(G,net.pos,start,goal)
    p3,c3 = bellman(G,start,goal)
    p4,c4 = ga.solve(start,goal)
    p5,c5 = aco.solve(start,goal)

    draw(G,net.pos,p1,name,c1)