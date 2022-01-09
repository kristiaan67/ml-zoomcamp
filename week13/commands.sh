
kind create cluster
kubectl get service

kind load docker-image churn-model:v001
kubectl get service

kubectl apply -f deployment.yaml
kubectl get deployment
kubectl get pod

kubectl apply -f service.yaml 
kubectl get service

kubectl port-forward service/churn 9696:80

python predict-test.py 