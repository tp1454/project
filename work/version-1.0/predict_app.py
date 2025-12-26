import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import subprocess
import sys
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Fragments, Lipinski, rdMolDescriptors, rdFingerprintGenerator, ChemicalFeatures
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.EState import AtomTypes as EAtomTypes
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

try:
    from rdkit.Chem import Draw
    HAS_DRAW = True
except ImportError:
    HAS_DRAW = False
    print("⚠️ Server thiếu thư viện vẽ hình. Đã tắt tính năng hiển thị cấu trúc.")

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

MORGAN_BITS = 512
MORGAN_RADIUS = 2
USE_MACCS = True
COMPUTE_3D = True
MAX_ITERS_3D = 0  
try:
    FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef')
except:
    FEATURE_FACTORY = None

def _safe(fn, default=0.0):
    def wrap(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return default
    return wrap

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        return Chem.MolToSmiles(mol, canonical=True)
    except: return None

def _count_atoms(m, symbols):
    if m is None: return 0
    s = set(symbols)
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() in s)

def _largest_ring_size(m):
    if m is None: return 0
    ri = m.GetRingInfo()
    return max((len(r) for r in ri.AtomRings()), default=0)

def count_explicit_h(m):
    if m is None: return 0
    mH = Chem.AddHs(m)
    return sum(1 for a in mH.GetAtoms() if a.GetSymbol() == 'H')

def gasteiger_stats(m):
    if m is None: return {'Gasteiger_q_sum': 0.0, 'Gasteiger_q_abs_sum': 0.0, 'Gasteiger_q_min': 0.0, 'Gasteiger_q_max': 0.0, 'Gasteiger_q_std': 0.0}
    mH = Chem.AddHs(m)
    try: AllChem.ComputeGasteigerCharges(mH)
    except: return {'Gasteiger_q_sum': 0.0, 'Gasteiger_q_abs_sum': 0.0, 'Gasteiger_q_min': 0.0, 'Gasteiger_q_max': 0.0, 'Gasteiger_q_std': 0.0}
    vals = []
    for a in mH.GetAtoms():
        try: v = float(a.GetProp('_GasteigerCharge')) if a.HasProp('_GasteigerCharge') else 0.0
        except: v = 0.0
        if pd.isna(v) or v in (float('inf'), float('-inf')): v = 0.0
        vals.append(v)
    arr = np.asarray(vals, dtype=float)
    return {'Gasteiger_q_sum': float(arr.sum()), 'Gasteiger_q_abs_sum': float(np.abs(arr).sum()), 'Gasteiger_q_min': float(arr.min(initial=0.0)), 'Gasteiger_q_max': float(arr.max(initial=0.0)), 'Gasteiger_q_std': float(arr.std(ddof=0))}

def _smiles_morphology(smi):
    if not smi: return {'SMI_len': 0, 'SMI_branches': 0, 'SMI_ringDigits': 0, 'SMI_stereoAt': 0, 'SMI_ezSlashes': 0}
    return {'SMI_len': len(smi), 'SMI_branches': smi.count('('), 'SMI_ringDigits': sum(ch.isdigit() for ch in smi), 'SMI_stereoAt': smi.count('@'), 'SMI_ezSlashes': smi.count('/') + smi.count('\\')}

def _estate_stats(m):
    if m is None: return {'EState_sum': 0.0, 'EState_mean': 0.0, 'EState_max': 0.0, 'EState_min': 0.0, 'EState_std': 0.0}
    try:
        vals = EAtomTypes.EStateIndices(m)
        if not vals: return {'EState_sum': 0.0, 'EState_mean': 0.0, 'EState_max': 0.0, 'EState_min': 0.0, 'EState_std': 0.0}
        arr = np.asarray(vals, dtype=float)
        return {'EState_sum': float(arr.sum()), 'EState_mean': float(arr.mean()), 'EState_max': float(arr.max()), 'EState_min': float(arr.min()), 'EState_std': float(arr.std(ddof=0))}
    except: return {'EState_sum': 0.0, 'EState_mean': 0.0, 'EState_max': 0.0, 'EState_min': 0.0, 'EState_std': 0.0}

def _bond_order(b):
    if b.GetIsAromatic(): return 1.5
    t = b.GetBondType()
    if t == Chem.BondType.SINGLE: return 1.0
    if t == Chem.BondType.DOUBLE: return 2.0
    if t == Chem.BondType.TRIPLE: return 3.0
    return 0.0

def _ring_size_hist(m):
    if m is None: return {5: 0, 6: 0, 7: 0, 8: 0}, 0
    ri = m.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    out = {5: 0, 6: 0, 7: 0, 8: 0}
    for s in sizes: 
        if s in out: out[s] += 1
    return out, len(sizes)

def _ring_systems_count(m):
    if m is None: return 0
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings: return 0
    seen = set()
    systems = 0
    for i in range(len(rings)):
        if i in seen: continue
        systems += 1
        stack = [i]; seen.add(i)
        while stack:
            j = stack.pop()
            for k in range(len(rings)):
                if k in seen: continue
                if rings[j] & rings[k]: seen.add(k); stack.append(k)
    return systems

try: from rdkit.Chem.Scaffolds import MurckoScaffold
except: MurckoScaffold = None

def _murcko_stats(m):
    if m is None or MurckoScaffold is None: return {'MurckoAtoms': 0, 'MurckoRings': 0, 'MurckoRingSystems': 0, 'SideChainAtoms': 0 if m is None else m.GetNumAtoms()}
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is None or scaf.GetNumAtoms() == 0: return {'MurckoAtoms': 0, 'MurckoRings': 0, 'MurckoRingSystems': 0, 'SideChainAtoms': m.GetNumAtoms()}
        return {'MurckoAtoms': int(scaf.GetNumAtoms()), 'MurckoRings': int(rdMolDescriptors.CalcNumRings(scaf)), 'MurckoRingSystems': int(_ring_systems_count(scaf)), 'SideChainAtoms': int(max(m.GetNumAtoms() - scaf.GetNumAtoms(), 0))}
    except: return {'MurckoAtoms': 0, 'MurckoRings': 0, 'MurckoRingSystems': 0, 'SideChainAtoms': m.GetNumAtoms()}

def _intramol_hbond_stats(m, embed_3d=False, maxIters=50, dist_cutoff=3.2, max_topo_path=6):
    default = {'IntraHBond_topo': 0, 'IntraHBond_minPath': 0, 'IntraHBond_3D': 0, 'IntraHBond_minDist3D': 0.0, 'IntraHBond_pairs': 0}
    if m is None or FEATURE_FACTORY is None: return default
    try:
        feats = FEATURE_FACTORY.GetFeaturesForMol(m)
        donors = [f.GetAtomIds()[0] for f in feats if f.GetFamily() == 'Donor']
        acceptors = [f.GetAtomIds()[0] for f in feats if f.GetFamily() == 'Acceptor']
        if not donors or not acceptors: return default
        topo = Chem.GetDistanceMatrix(m)
        paths = [topo[d, a] for d in donors for a in acceptors if d != a]
        min_path = float(np.min(paths)) if paths else 0.0
        topo_possible = int(min_path > 0 and min_path <= max_topo_path)
        out = default.copy()
        out.update({'IntraHBond_topo': topo_possible, 'IntraHBond_minPath': float(min_path), 'IntraHBond_pairs': float(len(paths))})
        if embed_3d:
            m3d = Chem.AddHs(Chem.Mol(m))
            params = AllChem.ETKDGv3() if hasattr(AllChem, 'ETKDGv3') else AllChem.ETKDG()
            params.randomSeed = 123; params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(m3d, params)
            if cid >= 0 and maxIters > 0: AllChem.UFFOptimizeMolecule(m3d, confId=cid, maxIters=maxIters)
            conf = m3d.GetConformer(); coords = conf.GetPositions(); dists = []
            for d in donors:
                for a in acceptors:
                    if d == a: continue
                    dists.append(float(np.linalg.norm(coords[d] - coords[a])))
            if dists:
                min_dist = float(min(dists))
                out['IntraHBond_minDist3D'] = min_dist
                out['IntraHBond_3D'] = int(min_dist <= dist_cutoff)
        return out
    except: return default

def augment_extra_cheaps(row, m):
    if m is None:
        row.update({'FracSingle': 0.0, 'FracDouble': 0.0, 'FracTriple': 0.0, 'FracAromatic': 0.0, 'MeanBondOrder': 0.0, 'UnsatBondCount': 0, 'Rings5': 0, 'Rings6': 0, 'Rings7': 0, 'Rings8': 0, 'RingSystems': 0, 'Rings56_frac': 0.0, 'FormalCharge': 0, 'IsZwitterion': 0})
        row.update(_estate_stats(m)); row.update(_murcko_stats(m)); row.update(_smiles_morphology(''))
        return row
    row.update(_estate_stats(m))
    bonds = list(m.GetBonds()); nb = max(len(bonds), 1)
    n_single = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic())
    n_double = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE)
    n_triple = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.TRIPLE)
    n_arom = sum(1 for b in bonds if b.GetIsAromatic())
    row['FracSingle'] = n_single / nb; row['FracDouble'] = n_double / nb
    row['FracTriple'] = n_triple / nb; row['FracAromatic'] = n_arom / nb
    row['MeanBondOrder'] = (sum(_bond_order(b) for b in bonds) / nb) if nb else 0.0
    row['UnsatBondCount'] = int(n_double + n_triple + n_arom)
    hist, n_rings = _ring_size_hist(m)
    row['Rings5'] = int(hist[5]); row['Rings6'] = int(hist[6]); row['Rings7'] = int(hist[7]); row['Rings8'] = int(hist[8])
    row['RingSystems'] = int(_ring_systems_count(m)); row['Rings56_frac'] = (hist[5] + hist[6]) / (n_rings if n_rings > 0 else 1)
    row.update(_murcko_stats(m))
    tot_charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
    has_pos = any(a.GetFormalCharge() > 0 for a in m.GetAtoms())
    has_neg = any(a.GetFormalCharge() < 0 for a in m.GetAtoms())
    row['FormalCharge'] = int(tot_charge); row['IsZwitterion'] = int(has_pos and has_neg)
    try: smi = Chem.MolToSmiles(m, canonical=True)
    except: smi = ''
    row.update(_smiles_morphology(smi))
    return row

def _shape3d_from_cansmi(cansmi, maxIters=0):
    try:
        m = Chem.MolFromSmiles(cansmi)
        if m is None: return {}
        mH = Chem.AddHs(m)
        params = AllChem.ETKDGv3() if hasattr(AllChem, 'ETKDGv3') else AllChem.ETKDG()
        params.randomSeed = 123; params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mH, params)
        if cid < 0: cid = AllChem.EmbedMolecule(mH, randomSeed=123)
        if cid < 0: return {}
        if maxIters > 0:
            try: AllChem.UFFOptimizeMolecule(mH, confId=cid, maxIters=int(maxIters))
            except: pass
        m_noH = Chem.RemoveHs(mH); out = {}
        for nm, fn in [('RadiusOfGyration', rdMolDescriptors.CalcRadiusOfGyration), ('InertialShapeFactor', rdMolDescriptors.CalcInertialShapeFactor), ('PMI1', rdMolDescriptors.CalcPMI1), ('PMI2', rdMolDescriptors.CalcPMI2), ('PMI3', rdMolDescriptors.CalcPMI3), ('NPR1', rdMolDescriptors.CalcNPR1), ('NPR2', rdMolDescriptors.CalcNPR2)]:
            try: out[nm] = float(fn(m_noH, confId=0))
            except: out[nm] = 0.0
        pmi1 = out.get('PMI1', 0.0) or 0.0
        out['PMI2_over_PMI1'] = (out.get('PMI2', 0.0) / pmi1) if pmi1 else 0.0
        out['PMI3_over_PMI1'] = (out.get('PMI3', 0.0) / pmi1) if pmi1 else 0.0
        return out
    except: return {}

def rdkit_feature_row_single(smiles, compute_3d=True):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None: return None
        row = {}
        for name, func in Descriptors._descList: row[name] = _safe(func, np.nan)(m)
        row['MolLogP'] = _safe(Crippen.MolLogP, np.nan)(m)
        row['MolMR'] = _safe(Crippen.MolMR, np.nan)(m)
        row['NumHAcceptors'] = _safe(Lipinski.NumHAcceptors, np.nan)(m)
        row['NumHDonors'] = _safe(Lipinski.NumHDonors, np.nan)(m)
        for fn_name, fn in [('NumRings', rdMolDescriptors.CalcNumRings), ('NumAromaticRings', rdMolDescriptors.CalcNumAromaticRings), ('NumAliphaticRings', rdMolDescriptors.CalcNumAliphaticRings), ('NumSaturatedRings', rdMolDescriptors.CalcNumSaturatedRings), ('NumBridgeheadAtoms', rdMolDescriptors.CalcNumBridgeheadAtoms), ('NumSpiroAtoms', rdMolDescriptors.CalcNumSpiroAtoms), ('NumAmideBonds', rdMolDescriptors.CalcNumAmideBonds), ('TPSA', rdMolDescriptors.CalcTPSA), ('LabuteASA', rdMolDescriptors.CalcLabuteASA)]:
            row[fn_name] = _safe(fn, 0)(m)
        row['LargestRingSize'] = _safe(_largest_ring_size, 0)(m)
        for el in ['C','N','O','S','F','Cl','Br','I','P']: row[f'Count_{el}'] = _count_atoms(m, [el])
        row['Count_H'] = _safe(count_explicit_h, 0)(m)
        for attr in dir(Fragments):
            if attr.startswith('fr_'):
                fn = getattr(Fragments, attr); 
                if callable(fn): row[attr] = _safe(fn, 0)(m)
        mgen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, countSimulation=False)
        mfp = mgen.GetFingerprint(m)
        for i in range(MORGAN_BITS): row[f'Morgan_{i}'] = int(mfp[i])
        if USE_MACCS:
            maccs = GenMACCSKeys(m)
            for i in range(len(maccs)): row[f'MACCS_{i}'] = int(maccs[i])
        for vsa_name, vsa_fn in [('SlogP_VSA', getattr(rdMolDescriptors, 'SlogP_VSA_', None)), ('SMR_VSA', getattr(rdMolDescriptors, 'SMR_VSA_', None)), ('EState_VSA', getattr(rdMolDescriptors, 'EState_VSA_', None))]:
            if vsa_fn:
                try:
                    bins = vsa_fn(m)
                    for i, val in enumerate(bins): row[f'{vsa_name}{i}'] = float(val)
                    row[f'{vsa_name}_sum'] = float(sum(bins))
                except: pass
        row.update(_safe(gasteiger_stats, {})(m))
        if compute_3d:
            try:
                cansmi = Chem.MolToSmiles(m, canonical=True)
                shape_feats = _shape3d_from_cansmi(cansmi, maxIters=MAX_ITERS_3D)
                row.update(shape_feats)
            except: pass
        hbd = float(row.get('NumHDonors', 0.0) or 0.0); hba = float(row.get('NumHAcceptors', 0.0) or 0.0)
        mw = float(row.get('MolWt', 0.0) or 0.0); hat = float(row.get('HeavyAtomCount', 0.0) or 1.0)
        tpsa = float(row.get('TPSA', 0.0) or 0.0); nrot = float(row.get('NumRotatableBonds', 0.0) or 0.0)
        narm = float(row.get('NumAromaticRings', 0.0) or 0.0); mollogp = float(row.get('MolLogP', 0.0) or 0.0)
        bertz = float(row.get('BertzCT', 0.0) or 0.0)
        row['HBondCapacity'] = hbd + hba
        row['HBondDensity_perHeavyAtom'] = (hbd + hba) / hat
        row['RingDensity_perHeavyAtom'] = float(row.get('NumRings', 0.0) or 0.0) / hat
        row['HalogenCount'] = float(row.get('Count_F', 0) + row.get('Count_Cl', 0) + row.get('Count_Br', 0) + row.get('Count_I', 0))
        row['HeteroAtomFrac'] = float(row.get('Count_N', 0) + row.get('Count_O', 0) + row.get('Count_S', 0) + row.get('Count_P', 0)) / hat
        row['AromRingFrac'] = float(row.get('NumAromaticRings', 0.0) or 0.0) / float((row.get('NumRings', 1.0) or 1.0))
        row['HBond_Product'] = hbd * hba; row['LogP_div_TPSA'] = mollogp / (tpsa + 1.0)
        row['LogP_x_TPSA'] = mollogp * tpsa; row['Flexibility_Score'] = nrot / (mw + 1.0)
        row['MolWt_x_AromaticRings'] = mw * narm; row['Complexity_per_MW'] = bertz / (mw + 1.0)
        row['Rigidity_Score'] = narm / (nrot + 1.0)
        row.update(_intramol_hbond_stats(m, embed_3d=compute_3d, maxIters=MAX_ITERS_3D))
        row = augment_extra_cheaps(row, m)
        return row
    except Exception as e:
        st.error(f"Lỗi tính features: {e}")
        return None

st.set_page_config(page_title="Dự đoán Tm", page_icon="🔥", layout="wide")
st.markdown("""
<style>
    .main-header {font-size: 30px; font-weight: bold; color: #FF4B4B; text-align: center;}
    .result-box {padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;}
    .result-value {font-size: 40px; font-weight: bold; color: #00CC00;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources(method):
    current_dir = Path(__file__).parent.absolute()
    base_path = current_dir / 'result'
    
    paths = {
        "GA": ("melting_point_model_ga.pkl", "features_list_ga.pkl"),
        "RFECV": ("melting_point_model_rfecv.pkl", "features_list_rfecv.pkl"),
        "Union": ("melting_point_model_uni.pkl", "features_list_uni.pkl"),
        "Intersection": ("melting_point_model_int.pkl", "features_list_int.pkl"),
    }
    
    m_file, f_file = paths.get(method)
    
    model_path = base_path / m_file
    feature_path = base_path / f_file

    try:
        model = joblib.load(model_path)
        features = joblib.load(feature_path)
        return model, features
    except FileNotFoundError:
        st.error(f"❌ Lỗi: Không tìm thấy file tại: {model_path}")
        st.write("📂 Các file hiện có trong thư mục result:")
        try:
            files_in_result = [f.name for f in base_path.glob('*')]
            st.write(files_in_result)
        except:
            st.write("Không đọc được thư mục result.")
        return None, None

with st.sidebar:
    st.title("Cấu hình")
    method = st.radio("Phương pháp FS:", ("GA", "RFECV", "Union", "Intersection"), index=2)
    st.info("**Nhóm 2 - Khai phá dữ liệu**\n\nGVHD: PGS. TS. Đỗ Như Tài")

st.markdown('<div class="main-header">DỰ ĐOÁN NHIỆT ĐỘ NÓNG CHẢY (Tm)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🧪 Nhập liệu")
    
    input_mode = st.radio("Nguồn dữ liệu:", ["Nhập thủ công (SMILES)", "Chọn chất mẫu có sẵn"])
    
    smiles_input = ""
    
    if input_mode == "Nhập thủ công (SMILES)":
        smiles_input = st.text_input("Nhập chuỗi SMILES:", placeholder="Ví dụ: CCO")
    else:
        s_dict = {
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Metformin": "CN(C)C(=N)NC(=N)N",
            "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
        }
        sel = st.selectbox("Chọn chất:", list(s_dict.keys()))
        if sel:
            smiles_input = s_dict[sel]
            st.info(f"SMILES: `{smiles_input}`")
            
    btn = st.button("🚀 Dự đoán", type="primary", use_container_width=True)

if btn and smiles_input:
    clean_smiles = standardize_smiles(smiles_input)
    
    if not clean_smiles:
        st.error("SMILES không hợp lệ hoặc lỗi cú pháp.")
    else:
        model, required_feats = load_resources(method)
        if not model:
            st.error(f"Không tìm thấy model {method}. Kiểm tra thư mục result/.")
        else:
            with st.spinner("Đang tính toán đặc trưng (gồm 3D shape)..."):
                full_feats_dict = rdkit_feature_row_single(clean_smiles, compute_3d=COMPUTE_3D)
                
            if full_feats_dict:
                input_data = []
                missing_feats = []
                for f in required_feats:
                    if f in full_feats_dict:
                        input_data.append(full_feats_dict[f])
                    else:
                        input_data.append(0.0)
                        missing_feats.append(f)
                
                X_input = pd.DataFrame([input_data], columns=required_feats)
                X_input = X_input.fillna(0)
                
                try:
                    pred_log = model.predict(X_input)[0]
                    pred_val = np.expm1(pred_log)
                    
                    with col2:
                        st.markdown("### 🎯 Kết quả")
                        if HAS_DRAW:
                            try:
                                mol = Chem.MolFromSmiles(clean_smiles)
                                if mol:
                                    st.image(Draw.MolToImage(mol), caption="Cấu trúc 2D", width=300)
                            except:
                                st.info("Không thể tạo hình ảnh cấu trúc.")
                        else:
                            st.warning("Server thiếu thư viện đồ họa nên không hiển thị hình ảnh.")
                        
                        tm_k = pred_val
                        tm_c = tm_k - 273.15
                        tm_f = (tm_c * 9/5) + 32

                        st.markdown("#### 🌡️ Nhiệt độ dự báo:")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric(label="Kelvin (K)", value=f"{tm_k:.2f}")
                        with m_col2:
                            st.metric(label="Celsius (°C)", value=f"{tm_c:.2f}")
                        with m_col3:
                            st.metric(label="Fahrenheit (°F)", value=f"{tm_f:.2f}")

                        st.write("")
                        if tm_c < 0:
                            st.info(f"❄️ Chất này có nhiệt độ nóng chảy thấp (Lỏng/Khí ở đk thường).")
                        elif tm_c < 100:
                            st.success(f"💧 Chất rắn dễ nóng chảy (tương đương sáp nến/bơ).")
                        elif tm_c < 300:
                            st.warning(f"🔥 Chất rắn nóng chảy trung bình.")
                        else:
                            st.error(f"🌋 Chất rắn chịu nhiệt cao.")
                        
                        if missing_feats:
                            st.warning(f"⚠️ Cảnh báo: Có {len(missing_feats)} đặc trưng thiếu.")
                            with st.expander("Chi tiết"): st.write(missing_feats)
                                
                except Exception as e:
                    st.error(f"Lỗi dự đoán: {e}")