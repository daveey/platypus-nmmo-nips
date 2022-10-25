cp $(ls -t results/$@/model* | head -n 1) checkpoints/$@.pt
git add checkpoints/$@.pt
git commit -a -m "add a checkpoint for $@.pt"
git push origin 
