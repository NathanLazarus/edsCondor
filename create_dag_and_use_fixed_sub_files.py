# create_dag_and_use_fixed_sub_files.py

'''this is neater, because you could change maxiter and n_jobs
in the same place and avaoid having a fit_init.sub file,
but without a way to set the data_file_names macro *once* globally,
the .dag file become unwieldy
'''

maxiter = 12
n_jobs = 3

jobs = [
    job_type + str(iter_num)
    for iter_num in range(maxiter + 1)
    for job_type in ["fit", "sim"]
]
jobs = jobs[:-1]  # don't need one last sim at the end

subfiles = [
    job_type + ".sub" for iter_num in range(maxiter + 1) for job_type in ["fit", "sim"]
][:-1]
# subfiles[0] = "fit_init.sub"

subsection = "".join("Job " + i + " " + j + "\n" for i, j in zip(jobs, subfiles))

res = "".join("PARENT " + i + " CHILD " + j + "\n" for i, j in zip(jobs[:-1], jobs[1:]))

fitjobs = [
    job
    for job in jobs
    if 'fit' in job
]

data_file_names = ", ".join("data" + str(i) + ".npy" for i in range(n_jobs))
hm = "".join("VARS " + i + " data_file_names=\"" + data_file_names + "\"\n" for i in fitjobs[1:])
simjobs = [
    job
    for job in jobs
    if 'sim' in job
]
hm2 = "".join("VARS " + i + " n_jobs=\"" + str(n_jobs) + "\"\n" for i in simjobs)

print(
    "# eds.dag\n\n" + subsection + "\n" + res + "\n" + 'VARS fit0 initialize="initialize"' + "\n" + hm + "\n" + hm2
)
