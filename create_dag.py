# create_dag.py

maxiter = 12
jobs = [
    job_type + str(iter_num)
    for iter_num in range(maxiter + 1)
    for job_type in ["fit", "sim"]
]
jobs = jobs[:-1]  # don't need one last sim at the end

subfiles = [
    job_type + ".sub" for iter_num in range(maxiter + 1) for job_type in ["fit", "sim"]
][:-1]
subfiles[0] = "fit_init.sub"

subsection = "".join("Job " + i + " " + j + "\n" for i, j in zip(jobs, subfiles))

dag_structure = "".join(
    "PARENT " + i + " CHILD " + j + "\n" for i, j in zip(jobs[:-1], jobs[1:])
)


eds_dag_file = "# eds.dag\n\n" + subsection + "\n" + dag_structure

with open("eds.dag", "w") as dag_file:
    dag_file.write(eds_dag_file)
