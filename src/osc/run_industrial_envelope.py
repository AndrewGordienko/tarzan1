from .packcell.industrial_envelope import evaluate
import argparse,json
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='artifacts/industrial_envelope_dev.json');a=ap.parse_args();r=evaluate();open(a.out,'w').write(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
