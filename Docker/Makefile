run:
	docker build -t distillation .

in:
	nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app --rm -it distillation:latest bash
