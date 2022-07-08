from lhotse.recipes import download_soapies, prepare_soapies

download_soapies("sandbox/root/soapies", languages="xhosa")
manifest = prepare_soapies("sandbox/root/soapies", languages="xhosa")

import pdb; pdb.set_trace()
