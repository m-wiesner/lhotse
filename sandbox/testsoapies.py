from lhotse.recipes import download_soapies, prepare_soapies

download_soapies("/expscratch/mwiesner/jsalt2022/data/soapies", languages="xhosa")
manifest = prepare_soapies("/expscratch/mwiesner/jsalt2022/data/soapies", output_dir="/expscratch/mwiesner/jsalt2022/lhotse/soapies_manifests", languages="xhosa")

import pdb; pdb.set_trace()
