
from mutation.argname_space import ArgNameSpace

AS = ArgNameSpace()

def find_sim_api_arg(api_name, arg_name):
    """ Returns an API name sampled from the full API list based on similarities. """
    return AS.find_sim(api_name, arg_name)
