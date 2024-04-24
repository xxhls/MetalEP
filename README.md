# 蛋白质金属结合位点致病性预测模型`MetalEP`

## 数据预处理

## 特征提取

ESM: `https://github.com/facebookresearch/esm`  

```bash
python scripts/extract_esm.py esm1v_t33_650M_UR90S_1 ./data/data_final.fasta ./features/esm --repr_layers 0 32 33 --include mean per_tok
```

ProtTrans: `https://github.com/agemagician/ProtTrans`

```bash
python scripts/extract_prot.py --input data/data_final.fasta --output features/prot/data_final.h5
```
