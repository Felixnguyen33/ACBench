from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import math
from auto_gptq import AutoGPTQForCausalLM  # For GPTQ models
import awq  # For AWQ models
from awq.models.base import BaseAWQForCausalLM  # For AWQ models

# R input N*d
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R

def cal_cov(R):
    with torch.no_grad():
        Z = torch.nn.functional.normalize(R, dim=1)
        A = torch.matmul(Z.T, Z) / Z.shape[0]
    return A

def cal_erank(A):
    with torch.no_grad():
        eig_val = torch.svd(A / torch.trace(A))[1]
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
    return erank

def compute_erank(R):
    return cal_erank(cal_cov(normalize(R)))

# 25.91 

model_path = "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct"
awq_path = "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ-W4-G128"
gptq_path = "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16"

tokenizer = AutoTokenizer.from_pretrained(model_path)
text = (
    "We introduce a rank-based metric called Diff-eRank, which is rooted in information "
    "theory and geometry principles. Diff-eRank evaluates LLMs by examining their hidden "
    "representations to quantify how LLMs discard redundant information after training."
)
inputs = tokenizer(text, return_tensors="pt")

# Process original model
model = AutoModel.from_pretrained(model_path).to('cuda:0')
with torch.no_grad():
    R_orig = model(inputs.input_ids.to('cuda:0'))[0][0, :, :]
    erank_orig = compute(R_orig)
del model
torch.cuda.empty_cache()

# Process AWQ model
awq_model = BaseAWQForCausalLM.from_pretrained(
    awq_path,
    trust_remote_code=True,
    use_cache=False
).to('cuda:0')
with torch.no_grad():
    R_awq = awq_model(inputs.input_ids.to('cuda:0'))[0][0, :, :]
    erank_awq = compute(R_awq)
    RD_awq = erank_awq - erank_orig
del awq_model
torch.cuda.empty_cache()

# Process GPTQ model
gptq_model = AutoGPTQForCausalLM.from_pretrained(
    gptq_path,
    trust_remote_code=True,
    use_cache=False
).to('cuda:0')
with torch.no_grad():
    R_gptq = gptq_model(inputs.input_ids.to('cuda:0'))[0][0, :, :]
    erank_gptq = compute(R_gptq)
    RD_gptq = erank_gptq - erank_orig
del gptq_model
torch.cuda.empty_cache()

print(f"AWQ Diff-eRank: {RD_awq}")
print(f"GPTQ Diff-eRank: {RD_gptq}")

