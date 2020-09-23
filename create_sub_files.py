# create_sub_files.py

n_jobs = 6


data_file_names = ", ".join("data" + str(i) + ".npy" for i in range(n_jobs))


fit_sub = """\
# fit.sub

universe                = docker
docker_image            = nathanlazarus/mypython
executable              = python
transfer_executable     = False
arguments               = fit.py
transfer_input_files    = fit.py, {}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
output                  = outfiles/fit$(Cluster).out
error                   = outfiles/fit$(Cluster).err
log                     = outfiles/edsCondor.log
queue
""".format(data_file_names)

with open("fit.sub", "w") as sub_file:
    sub_file.write(fit_sub)



fit_init_sub = """\
# fit_init.sub

universe                = docker
docker_image            = nathanlazarus/mypython
executable              = python
transfer_executable     = False
arguments               = fit.py initialize
transfer_input_files    = fit.py
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
output                  = outfiles/fit$(Cluster).out
error                   = outfiles/fit$(Cluster).err
log                     = outfiles/edsCondor.log
queue
""".format(data_file_names)

with open("fit_init.sub", "w") as sub_file:
    sub_file.write(fit_init_sub)



sim_sub = """\
# sim.sub

universe                = docker
docker_image            = nathanlazarus/mypython
executable              = python
transfer_executable     = False
arguments               = sim.py $(Process)
transfer_input_files    = sim.py, coefs.npy
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
output                  = outfiles/sim$(Cluster)_$(Process).out
error                   = outfiles/sim$(Cluster)_$(Process).err
log                     = outfiles/edsCondor.log
queue {}
""".format(n_jobs)

with open("sim.sub", "w") as sub_file:
    sub_file.write(sim_sub)
