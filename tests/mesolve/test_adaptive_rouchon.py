import pytest

from dynamiqs.gradient import Direct
from dynamiqs.method import Rouchon2, Rouchon3, Rouchon4, Rouchon5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - exact_expm: set to False
# - skip system dia_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveRouchon(IntegratorTester):
    @pytest.mark.parametrize('method_class', [Rouchon2, Rouchon3, Rouchon4, Rouchon5])
    @pytest.mark.parametrize(
        ('system', 'time_independent'), [(dense_ocavity, True), (otdqubit, False)]
    )
    def test_correctness(self, method_class, system, time_independent):
        self._test_correctness(system, method_class(time_independent=time_independent))

    @pytest.mark.parametrize('method_class', [Rouchon2, Rouchon3, Rouchon4, Rouchon5])
    @pytest.mark.parametrize(
        ('system', 'time_independent'), [(dense_ocavity, True), (otdqubit, False)]
    )
    @pytest.mark.parametrize('gradient', [Direct()])
    def test_gradient(self, method_class, system, time_independent, gradient):
        self._test_gradient(
            system, method_class(time_independent=time_independent), gradient
        )
