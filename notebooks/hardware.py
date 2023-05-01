import qiskit_braket_provider
from qiskit import QuantumCircuit
from qiskit_braket_provider import BraketLocalBackend
from matplotlib import pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_braket_provider import AWSBraketProvider

circuit = QuantumCircuit(3)
circuit.h(0)
for qubit in range(1, 3):
    circuit.cx(0, qubit)

# circuit.draw('mpl')
# plt.show()

# local_simulator = BraketLocalBackend()
# task = local_simulator.run(circuit, shots=1000)

# plot_histogram(task.result().get_counts())
# plt.show()

provider = AWSBraketProvider()
ionq_device = provider.get_backend("IonQ Device")
# rigetti_device = provider.get_backend("Aspen-M-1")
# oqc_device = provider.get_backend("Lucy")

ionq_task = ionq_device.run(circuit, shots=100)

ionq_arn = ionq_task.job_id()

ionq_retrieved = ionq_device.retrieve_job(job_id=ionq_arn)

ionq_retrieved.status()
