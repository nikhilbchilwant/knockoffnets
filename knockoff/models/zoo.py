import knockoff.models.classification #Though IDE shows that this import is not used, it is used by get_net

#Fetch relevant model
def get_net(modelname, modeltype, pretrained=None, **kwargs):
	assert modeltype in ('classification', 'entailment', 'sentiment')
	model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(**kwargs)
	return model