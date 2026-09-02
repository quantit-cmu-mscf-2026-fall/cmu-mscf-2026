"""
Approximate PJM pnode coordinates via PJM's own zip-code mapping + free zip geocode.
No paid geocoder. Coords are zip centroids -> approximate, incomplete (CEII: PJM
does not publish real coordinates or zone shapefiles). Good for regional viz only.

Run locally. Needs: pandas, requests, matplotlib, xlrd (for the .xls). 
Optional geographic zips file auto-downloaded from a public source.
"""
import io, os, requests
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# --- 1. PJM's official pnode -> zip-code mapping (from LMP Model Info) ---
ZIP_MAP_URL = "https://www.pjm.com/-/media/markets-ops/energy/lmp-model-info/zip-code-mapping.ashx"

def load_pjm_zip_mapping():
    r = requests.get(ZIP_MAP_URL, timeout=60); r.raise_for_status()
    # PJM file is .xls with ~9 header rows before the table (per pvvm repo)
    df = pd.read_excel(io.BytesIO(r.content), skiprows=9, dtype={"Zip Code": str})
    df.columns = [c.strip() for c in df.columns]
    df["Zip Code"] = df["Zip Code"].str.zfill(5)
    print("PJM zip-map columns:", list(df.columns))   # inspect: PNODEID/PNODENAME/Zip Code/Zone etc.
    return df

# --- 2. free zip -> lat/lon (public gist of US zip centroids) ---
ZIP_LATLON_URL = "https://raw.githubusercontent.com/millbj92/US-Zip-Codes-JSON/master/USCities.json"

def load_zip_latlon():
    r = requests.get(ZIP_LATLON_URL, timeout=60); r.raise_for_status()
    z = pd.read_json(io.BytesIO(r.content))
    # columns typically: zip_code, latitude, longitude, city, state, county
    z = z.rename(columns={"zip_code":"Zip Code","latitude":"lat","longitude":"lon"})
    z["Zip Code"] = z["Zip Code"].astype(str).str.zfill(5)
    return z[["Zip Code","lat","lon"]]

# --- 3. merge, then average multi-zip nodes to a single point ---
def build_node_coords():
    zmap = load_pjm_zip_mapping()
    zll  = load_zip_latlon()
    # normalize likely column names
    idcol   = next(c for c in zmap.columns if "PNODEID" in c.upper() or c.upper()=="ID")
    namecol = next((c for c in zmap.columns if "NAME" in c.upper()), idcol)
    zonecol = next((c for c in zmap.columns if "ZONE" in c.upper()), None)

    m = zmap.merge(zll, on="Zip Code", how="left")
    # cartesian avg avoids lon-wrap issues when a node maps to several zips
    m["x"] = np.cos(np.radians(m.lat))*np.cos(np.radians(m.lon))
    m["y"] = np.cos(np.radians(m.lat))*np.sin(np.radians(m.lon))
    m["z"] = np.sin(np.radians(m.lat))
    g = m.groupby([idcol]).agg(
        name=(namecol,"first"),
        zone=(zonecol,"first") if zonecol else (idcol,"first"),
        x=("x","mean"), y=("y","mean"), z=("z","mean"),
        n_zips=("Zip Code","nunique")).reset_index()
    g["lat"] = np.degrees(np.arcsin(g.z))
    g["lon"] = np.degrees(np.arctan2(g.y, g.x))
    g = g.dropna(subset=["lat","lon"]).rename(columns={idcol:"pnode_id"})
    return g[["pnode_id","name","zone","lat","lon","n_zips"]]

# --- 4. plot: nodes colored by zone + zone centroids ---
def plot(nodes):
    fig, ax = plt.subplots(figsize=(12,9))
    zones = nodes["zone"].astype(str)
    for z, sub in nodes.groupby(zones):
        ax.scatter(sub.lon, sub.lat, s=8, alpha=.5, label=z)
    cent = nodes.groupby(zones)[["lat","lon"]].mean()
    ax.scatter(cent.lon, cent.lat, c="k", marker="x", s=90)
    for z,row in cent.iterrows():
        ax.annotate(str(z),(row.lon,row.lat),fontsize=8,weight="bold")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("PJM pnodes (approx via zip centroids) — colored by zone")
    ax.legend(fontsize=6, ncol=2, markerscale=1.5, loc="lower left")
    plt.tight_layout(); plt.savefig("pjm_node_map.png", dpi=120)
    print("saved pjm_node_map.png")

if __name__ == "__main__":
    nodes = build_node_coords()
    print(f"\n{len(nodes):,} pnodes with approx coords; "
          f"{nodes['zone'].nunique()} zones")
    print(nodes.head(10).to_string())
    nodes.to_csv("pjm_node_coords_approx.csv", index=False)
    plot(nodes)