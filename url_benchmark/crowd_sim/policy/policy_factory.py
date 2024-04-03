from url_benchmark.crowd_sim.policy.linear import Linear
# from url_benchmark.crowd_sim.policy.orca import ORCA, CentralizedORCA
from url_benchmark.crowd_sim.policy.socialforce import SocialForce, CentralizedSocialForce


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
# policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
# policy_factory['centralized_orca'] = CentralizedORCA
policy_factory['centralized_socialforce'] = CentralizedSocialForce
policy_factory['none'] = none_policy
