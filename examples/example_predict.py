from optimum import Optimum
# if not in the same directory, must specify PTYHONPATH
optimum = Optimum("mymodel")
predict_model = optimum.load_model("gs://gnomondigital-sandbox-tvm-job-output/job_id=100004/optimized_model/", "llvm -mcpu cascadelake")
inputs_dict = {}
result = predict_model.predict(inputs_dict)
print(result)