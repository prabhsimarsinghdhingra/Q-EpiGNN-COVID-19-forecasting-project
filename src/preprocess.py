"""Preprocess JHU CSSE US COVID-19 data -> state-level spatiotemporal tensor + graph.

Input : data/raw/time_series_covid19_{confirmed,deaths}_US.csv  (JHU CSSE, county level)
Output: data/processed/covid_us_states.npz
"""
import numpy as np
import pandas as pd

RAW = "data/raw/"
OUT = "data/processed/"
STATES = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
 "District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky",
 "Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
 "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
 "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas",
 "Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
# Land borders between states (AK and HI are isolated; handled by a nearest-neighbour fallback below)
BORDERS = """AL-FL AL-GA AL-MS AL-TN AR-LA AR-MS AR-MO AR-OK AR-TN AR-TX AZ-CA AZ-NV AZ-NM AZ-UT AZ-CO CA-NV CA-OR
CO-KS CO-NE CO-NM CO-OK CO-UT CO-WY CT-MA CT-NY CT-RI DE-MD DE-NJ DE-PA DC-MD DC-VA FL-GA GA-NC GA-SC GA-TN
IA-MN IA-MO IA-NE IA-SD IA-WI IA-IL ID-MT ID-NV ID-OR ID-UT ID-WA ID-WY IL-IN IL-KY IL-MO IL-WI IN-KY IN-MI IN-OH
KS-MO KS-NE KS-OK KY-MO KY-OH KY-TN KY-VA KY-WV LA-MS LA-TX MA-NH MA-NY MA-RI MA-VT MD-PA MD-VA MD-WV ME-NH
MI-OH MI-WI MN-ND MN-SD MN-WI MO-NE MO-OK MO-TN MS-TN MT-ND MT-SD MT-WY NC-SC NC-TN NC-VA ND-SD NE-SD NE-WY
NH-VT NJ-NY NJ-PA NM-OK NM-TX NV-OR NV-UT NY-PA NY-VT OH-PA OH-WV OK-TX OR-WA PA-WV SD-WY TN-VA UT-WY VA-WV""".split()
ABBR = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut",
"DE":"Delaware","DC":"District of Columbia","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois",
"IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
"MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada",
"NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota",
"OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota",
"TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia",
"WI":"Wisconsin","WY":"Wyoming"}


def build_graph(cent, P):
    """Border adjacency A (N,N), gravity-weighted adjacency W, gravity matrix G (all symmetric).

    cent: (N,2) population-weighted centroids (lat, lon); P: (N,) populations. Order = STATES.
    """
    N = len(STATES)
    idx = {s: i for i, s in enumerate(STATES)}
    A = np.zeros((N, N))
    for b in BORDERS:
        a, c = b.split("-")
        A[idx[ABBR[a]], idx[ABBR[c]]] = A[idx[ABBR[c]], idx[ABBR[a]]] = 1
    for s in ["Alaska", "Hawaii"]:  # isolated states -> link to 2 nearest centroids (documented assumption)
        i = idx[s]
        d = np.linalg.norm(cent - cent[i], axis=1)
        d[i] = 1e9
        for j in np.argsort(d)[:2]:
            A[i, j] = A[j, i] = 1
    dist = np.linalg.norm(cent[:, None] - cent[None], axis=-1) + 1e-3
    G = (P[:, None] * P[None]) / dist ** 2  # gravity-type mobility proxy
    np.fill_diagonal(G, 0)
    W = A * G
    W = W / W.max()
    return A, W, G


def main():
    conf = pd.read_csv(RAW + "time_series_covid19_confirmed_US.csv")
    dth = pd.read_csv(RAW + "time_series_covid19_deaths_US.csv")
    dcols = [c for c in conf.columns if c.count("/") == 2]
    conf = conf[conf.Province_State.isin(STATES)]
    dth = dth[dth.Province_State.isin(STATES)]
    C = conf.groupby("Province_State")[dcols].sum().loc[STATES]
    D = dth.groupby("Province_State")[dcols].sum().loc[STATES]
    pop = dth.groupby("Province_State")["Population"].sum().loc[STATES]

    # population-weighted state centroid (lat, lon)
    w = dth[dth.Lat != 0].copy()
    w["la"] = w.Lat * w.Population
    w["lo"] = w.Long_ * w.Population
    g = w.groupby("Province_State")[["la", "lo", "Population"]].sum().loc[STATES]
    cent = np.stack([g.la / g.Population, g.lo / g.Population], 1)

    dates = pd.to_datetime(dcols, format="%m/%d/%y")
    newC = np.clip(np.diff(C.values, axis=1, prepend=0), 0, None).T.astype(float)  # (T,N) daily new cases
    newD = np.clip(np.diff(D.values, axis=1, prepend=0), 0, None).T.astype(float)
    keep = dates <= "2023-03-09"  # JHU CSSE stopped updating on 2023-03-10
    newC, newD, dates = newC[keep], newD[keep], dates[keep]

    # 7-day trailing moving average removes weekday reporting artefacts (causal: no look-ahead)
    def sm(x):
        return pd.DataFrame(x).rolling(7, min_periods=1).mean().values

    newC7, newD7 = sm(newC), sm(newD)

    P = pop.values.astype(float)
    A, W, G = build_graph(cent, P)

    np.savez_compressed(OUT + "covid_us_states.npz", cases=newC7, deaths=newD7, raw_cases=newC, pop=P,
                        centroid=cent, A=A, W=W, gravity=G / G.max(),
                        dates=dates.strftime("%Y-%m-%d").values, states=np.array(STATES))
    print("T,N =", newC7.shape, "| undirected edges:", int(A.sum() / 2),
          "| span:", dates[0].date(), "->", dates[-1].date())
    print("total confirmed:", int(newC.sum()), "| total deaths:", int(newD.sum()))


if __name__ == "__main__":
    main()
