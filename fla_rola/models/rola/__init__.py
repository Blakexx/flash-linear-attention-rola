
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla_rola.models.rola.configuration_rola import RoLAConfig
from fla_rola.models.rola.instances import ROLA_INSTANCES, rola_instance
from fla_rola.models.rola.modeling_rola import RoLAForCausalLM, RoLAModel

AutoConfig.register(RoLAConfig.model_type, RoLAConfig, exist_ok=True)
AutoModel.register(RoLAConfig, RoLAModel, exist_ok=True)
AutoModelForCausalLM.register(RoLAConfig, RoLAForCausalLM, exist_ok=True)


__all__ = ['ROLA_INSTANCES', 'RoLAConfig', 'RoLAForCausalLM', 'RoLAModel', 'rola_instance']
