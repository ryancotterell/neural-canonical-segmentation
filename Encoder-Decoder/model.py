# -*- coding: utf-8 -*-

from theano import tensor
from toolz import merge

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Random, Initializable, MLP, NDimensionalSoftmax, Softmax)
from blocks.bricks.attention import GenericSequenceAttention, ShallowEnergyComputer
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
     LookupFeedback, AbstractReadout, SoftmaxEmitter,AbstractEmitter)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans

from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.bricks.wrappers import WithExtraDims
from blocks.utils import dict_union, dict_subset, pack
from blocks.bricks.attention import (
    AbstractAttentionRecurrent, AttentionRecurrent)

from picklable_itertools.extras import equizip
  
	
class CostObject(Initializable):
  
  @lazy()
  def __init__(self, cost_type='original', **kwargs):
      super(CostObject, self).__init__(**kwargs)
      self.cost_type=cost_type
      self.softmax = Softmax()
      self.children = [self.softmax]
  
  @application(inputs=['input_'], outputs=['output'])
  def log_probabilities(self, input_):
      """Normalize log-probabilities.
      Converts unnormalized log-probabilities (exponents of which do not
      sum to one) into actual log-probabilities (exponents of which sum
      to one).
      Parameters
      ----------
      input_ : :class:`~theano.Variable`
          A matrix, each row contains unnormalized log-probabilities of a
          distribution.
      Returns
      -------
      output : :class:`~theano.Variable`
          A matrix with normalized log-probabilities in each row for each
          distribution from `input_`.
      """
      shifted = input_ - input_.max(axis=1, keepdims=True)
      return shifted - tensor.log(
          tensor.exp(shifted).sum(axis=1, keepdims=True))
      
  
  @application(inputs=['x', 'y'], outputs=['output'])
  def original_cost(self, x, y):
    x = self.log_probabilities(x)
    if y.ndim == x.ndim - 1:
        indices = tensor.arange(y.shape[0]) * x.shape[1] + y
        cost = -x.flatten()[indices]
    elif y.ndim == x.ndim:
        cost = -(x * y).sum(axis=1)
    else:
        raise TypeError('rank mismatch between x and y')
    return cost
  
  @application(inputs=['x', 'y'], outputs=['output'])
  def simple_cost(self, x, y):
    if y.ndim == x.ndim - 1:
        # Get probs:
        newX = self.softmax.apply(x)
        indices = tensor.arange(y.shape[0]) * x.shape[1] + y
        newY = tensor.ones_like(newX)
        cost = ((newY-newX).flatten()[indices])
    elif y.ndim == x.ndim:
        raise TypeError('\nExpected either x or y to be of another rank\n')
    else:
        raise TypeError('rank mismatch between x and y')
    return cost
    
  # x are the gold labels.
  @application(inputs=['x', 'y'], outputs=['output'])
  def cost(self, application_call, x, y):
    if self.cost_type=='original':
      return self.original_cost(x, y)
    if self.cost_type=='simple':
      return self.simple_cost(x,y)
    return 0
    
class NDimensionalCostObject(CostObject):
    decorators = [WithExtraDims()]
    
class BaseSequenceGenerator(Initializable):
    r"""A generic sequence generator.
    This class combines two components, a readout network and an
    attention-equipped recurrent transition, into a context-dependent
    sequence generator. Third component must be also given which
    forks feedback from the readout network to obtain inputs for the
    transition.
    The class provides two methods: :meth:`generate` and :meth:`cost`. The
    former is to actually generate sequences and the latter is to compute
    the cost of generating given sequences.
    The generation algorithm description follows.
    **Definitions and notation:**
    * States :math:`s_i` of the generator are the states of the transition
      as specified in `transition.state_names`.
    * Contexts of the generator are the contexts of the
      transition as specified in `transition.context_names`.
    * Glimpses :math:`g_i` are intermediate entities computed at every
      generation step from states, contexts and the previous step glimpses.
      They are computed in the transition's `apply` method when not given
      or by explicitly calling the transition's `take_glimpses` method. The
      set of glimpses considered is specified in
      `transition.glimpse_names`.
    * Outputs :math:`y_i` are produced at every step and form the output
      sequence. A generation cost :math:`c_i` is assigned to each output.
    **Algorithm:**
    1. Initialization.
       .. math::
           y_0 = readout.initial\_outputs(contexts)\\
           s_0, g_0 = transition.initial\_states(contexts)\\
           i = 1\\
       By default all recurrent bricks from :mod:`~blocks.bricks.recurrent`
       have trainable initial states initialized with zeros. Subclass them
       or :class:`~blocks.bricks.recurrent.BaseRecurrent` directly to get
       custom initial states.
    2. New glimpses are computed:
       .. math:: g_i = transition.take\_glimpses(
           s_{i-1}, g_{i-1}, contexts)
    3. A new output is generated by the readout and its cost is
       computed:
       .. math::
            f_{i-1} = readout.feedback(y_{i-1}) \\
            r_i = readout.readout(f_{i-1}, s_{i-1}, g_i, contexts) \\
            y_i = readout.emit(r_i) \\
            c_i = readout.cost(r_i, y_i)
       Note that the *new* glimpses and the *old* states are used at this
       step. The reason for not merging all readout methods into one is
       to make an efficient implementation of :meth:`cost` possible.
    4. New states are computed and iteration is done:
       .. math::
           f_i = readout.feedback(y_i) \\
           s_i = transition.compute\_states(s_{i-1}, g_i,
                fork.apply(f_i), contexts) \\
           i = i + 1
    5. Back to step 2 if the desired sequence
       length has not been yet reached.
    | A scheme of the algorithm described above follows.
    .. image:: /_static/sequence_generator_scheme.png
            :height: 500px
            :width: 500px
    ..
    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component of the sequence generator.
    transition : instance of :class:`AbstractAttentionRecurrent`
        The transition component of the sequence generator.
    fork : :class:`.Brick`
        The brick to compute the transition's inputs from the feedback.
    See Also
    --------
    :class:`.Initializable` : for initialization parameters
    :class:`SequenceGenerator` : more user friendly interface to this\
        brick
    """
    @lazy()
    def __init__(self, readout, transition, fork, cost_type, **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        self.readout = readout
        self.transition = transition
        self.fork = fork
        self.next_probs = 0.0
        self.cost_obj = None
        self.cost_type = cost_type
        if cost_type == 'categorical_cross_entropy':
          self.cost_obj = self.readout 
          self.children = [self.readout, self.fork, self.transition, self.readout.emitter]
        else:
          self.cost_obj = NDimensionalCostObject(self.cost_type)
          self.children = [self.readout, self.fork, self.transition, self.readout.emitter, self.cost_obj]
        assert self.cost_obj != None

    @property
    def _state_names(self):
        return self.transition.compute_states.outputs

    @property
    def _context_names(self):
        return self.transition.apply.contexts

    @property
    def _glimpse_names(self):
        return self.transition.take_glimpses.outputs

    def _push_allocation_config(self):
        self.transition.push_allocation_config()
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.readout.source_dims = [self.transition.get_dim(name)
                                    if name in transition_sources
                                    else self.readout.get_dim(name)
                                    for name in self.readout.source_names]

        # Configure fork.
        self.readout.push_allocation_config()
        feedback_name, = self.readout.feedback.outputs
        self.fork.input_dim = self.readout.get_dim(feedback_name)
        self.fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)

    @application
    def cost(self, application_call, outputs, mask=None, **kwargs):
        """Returns the average cost over the minibatch.
        The cost is computed by averaging the sum of per token costs for
        each sequence over the minibatch.
        .. warning::
            Note that, the computed cost can be problematic when batches
            consist of vastly different sequence lengths.
        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The 3(2) dimensional tensor containing output sequences.
            The axis 0 must stand for time, the axis 1 for the
            position in the batch.
        mask : :class:`~tensor.TensorVariable`
            The binary matrix identifying fake outputs.
        Returns
        -------
        cost : :class:`~tensor.Variable`
            Theano variable for cost, computed by summing over timesteps
            and then averaging over the minibatch.
        Notes
        -----
        The contexts are expected as keyword arguments.
        Adds average cost per sequence element `AUXILIARY` variable to
        the computational graph with name ``per_sequence_element``.
        """
        # Compute the sum of costs.
        costs = self.cost_matrix(outputs, mask=mask, **kwargs)
        cost = tensor.mean(costs.sum(axis=0))
        add_role(cost, COST)

        # Add auxiliary variable for per sequence element cost.
        application_call.add_auxiliary_variable(
            (costs.sum() / mask.sum()) if mask is not None else costs.mean(),
            name='per_sequence_element')
        return cost

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.
        See Also
        --------
        :meth:`cost` : Scalar cost.
        """
        # We assume the data has axes (time, batch, features, ...).
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part.
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # Masks in context are optional (e.g. `attended_mask`).
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network.
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost.
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
	if self.cost_type != 'categorical_cross_entropy':
          costs = self.cost_obj.cost(readouts, outputs, extra_ndim=readouts.ndim - 2)
        else:
	  costs = self.cost_obj.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs

    @application
    def getTestVar(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.
        See Also
        --------
        :meth:`cost` : Scalar cost.
        """
        # We assume the data has axes (time, batch, features, ...).
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part.
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # Masks in context are optional (e.g. `attended_mask`).
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network.
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost.
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
	
	return readouts, outputs
      
    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.
        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.
        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.
        """
        states = dict_subset(kwargs, self._state_names)
        # Masks in context are optional (e.g. `attended_mask`).
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names)

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs, self.next_probs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + [next_costs] + [self.next_probs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'] + [self.next_probs])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states
      
class Readout(AbstractReadout):
    r"""Readout brick with separated emitter and feedback parts.
    :class:`Readout` combines a few bits and pieces into an object
    that can be used as the readout component in
    :class:`BaseSequenceGenerator`. This includes an emitter brick,
    to which :meth:`emit`, :meth:`cost` and :meth:`initial_outputs`
    calls are delegated, a feedback brick to which :meth:`feedback`
    functionality is delegated, and a pipeline to actually compute
    readouts from all the sources (see the `source_names` attribute
    of :class:`AbstractReadout`).
    The readout computation pipeline is constructed from `merge` and
    `post_merge` brick, whose responsibilites are described in the
    respective docstrings.
    Parameters
    ----------
    emitter : an instance of :class:`AbstractEmitter`
        The emitter component.
    feedback_brick : an instance of :class:`AbstractFeedback`
        The feedback component.
    merge : :class:`.Brick`, optional
        A brick that takes the sources given in `source_names` as an input
        and combines them into a single output. If given, `merge_prototype`
        cannot be given.
    merge_prototype : :class:`.FeedForward`, optional
        If `merge` isn't given, the transformation given by
        `merge_prototype` is applied to each input before being summed. By
        default a :class:`.Linear` transformation without biases is used.
        If given, `merge` cannot be given.
    post_merge : :class:`.Feedforward`, optional
        This transformation is applied to the merged inputs. By default
        :class:`.Bias` is used.
    merged_dim : int, optional
        The input dimension of `post_merge` i.e. the output dimension of
        `merge` (or `merge_prototype`). If not give, it is assumed to be
        the same as `readout_dim` (i.e. `post_merge` is assumed to not
        change dimensions).
    \*\*kwargs : dict
        Passed to the parent's constructor.
    See Also
    --------
    :class:`BaseSequenceGenerator` : see how exactly a readout is used
    :class:`AbstractEmitter`, :class:`AbstractFeedback`
    """
    def __init__(self, emitter=None, feedback_brick=None,
                 merge=None, merge_prototype=None,
                 post_merge=None, merged_dim=None, **kwargs):
        super(Readout, self).__init__(**kwargs)

        if not emitter:
            emitter = TrivialEmitter(self.readout_dim)
        if not feedback_brick:
            feedback_brick = TrivialFeedback(self.readout_dim)
        if not merge:
            merge = Merge(input_names=self.source_names,
                          prototype=merge_prototype)
        if not post_merge:
            post_merge = Bias(dim=self.readout_dim)
        if not merged_dim:
            merged_dim = self.readout_dim
        self.emitter = emitter
        self.feedback_brick = feedback_brick
        self.merge = merge
        self.post_merge = post_merge
        self.merged_dim = merged_dim

        self.children = [self.emitter, self.feedback_brick,
                         self.merge, self.post_merge]

    def _push_allocation_config(self):
        self.emitter.readout_dim = self.get_dim('readouts')
        self.feedback_brick.output_dim = self.get_dim('outputs')
        self.merge.input_names = self.source_names
        self.merge.input_dims = self.source_dims
        self.merge.output_dim = self.merged_dim
        self.post_merge.input_dim = self.merged_dim
        self.post_merge.output_dim = self.readout_dim

    @application
    def readout(self, **kwargs):
        merged = self.merge.apply(**{name: kwargs[name]
                                     for name in self.merge.input_names})
        merged = self.post_merge.apply(merged)
        return merged

    @application
    def emit(self, readouts):
        return self.emitter.emit(readouts)

    @application
    def cost(self, readouts, outputs):
        return self.emitter.cost(readouts, outputs)

    @application
    def initial_outputs(self, batch_size):
        return self.emitter.initial_outputs(batch_size)

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.feedback_brick.feedback(outputs)

    def get_dim(self, name):
        if name == 'outputs':
            return self.emitter.get_dim(name)
        elif name == 'feedback':
            return self.feedback_brick.get_dim(name)
        elif name == 'readouts':
            return self.readout_dim
        return super(Readout, self).get_dim(name)
      
class SequenceGenerator(BaseSequenceGenerator):
    r"""A more user-friendly interface for :class:`BaseSequenceGenerator`.
    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : object, optional
        The attention mechanism to be added to ``transition``,
        an instance of
        :class:`~blocks.bricks.attention.AbstractAttention`.
    add_contexts : bool
        If ``True``, the
        :class:`.AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and its
        mask.
    \*\*kwargs : dict
        All keywords arguments are passed to the base class. If `fork`
        keyword argument is not provided, :class:`.Fork` is created
        that forks all transition sequential inputs without a "mask"
        substring in them.
    """
    def __init__(self, readout, transition, attention=None,
                 add_contexts=True, cost_type=None, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        if attention:
            transition = AttentionRecurrent(
                transition, attention,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        super(SequenceGenerator, self).__init__(
            readout, transition, cost_type=cost_type, **kwargs)
	
class SequenceContentAttention2(GenericSequenceAttention, Initializable):
    """Attention mechanism that looks for relevant content in a sequence.
    This is the attention mechanism used in [BCB]_. The idea in a nutshell:
    1. The states and the sequence are transformed independently,
    2. The transformed states are summed with every transformed sequence
       element to obtain *match vectors*,
    3. A match vector is transformed into a single number interpreted as
       *energy*,
    4. Energies are normalized in softmax-like fashion. The resulting
       summing to one weights are called *attention weights*,
    5. Weighted average of the sequence elements with attention weights
       is computed.
    In terms of the :class:`AbstractAttention` documentation, the sequence
    is the attended. The weighted averages from 5 and the attention
    weights from 4 form the set of glimpses produced by this attention
    mechanism.
    Parameters
    ----------
    state_names : list of str
        The names of the network states.
    attended_dim : int
        The dimension of the sequence elements.
    match_dim : int
        The dimension of the match vector.
    state_transformer : :class:`.Brick`
        A prototype for state transformations. If ``None``,
        a linear transformation is used.
    attended_transformer : :class:`.Feedforward`
        The transformation to be applied to the sequence. If ``None`` an
        affine transformation is used.
    energy_computer : :class:`.Feedforward`
        Computes energy from the match vector. If ``None``, an affine
        transformations preceeded by :math:`tanh` is used.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    .. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
       Machine Translation by Jointly Learning to Align and Translate.
    """
    @lazy(allocation=['match_dim'])
    def __init__(self, match_dim, state_transformer=None,
                 attended_transformer=None, energy_computer=None, **kwargs):
        super(SequenceContentAttention2, self).__init__(**kwargs)
        if not state_transformer:
            state_transformer = Linear(use_bias=False)
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            attended_transformer = Linear(name="preprocess")
        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.attended_transformer = attended_transformer
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, attended_transformer,
                         energy_computer]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application
    def compute_energies(self, attended, preprocessed_attended, states):
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_attended)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, **states):
        r"""Compute attention weights and produce glimpses.
        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.
        preprocessed_attended : :class:`~tensor.TensorVariable`
            The preprocessed sequence. If ``None``, is computed by calling
            :meth:`preprocess`.
        attended_mask : :class:`~tensor.TensorVariable`
            A 0/1 mask specifying available data. 0 means that the
            corresponding sequence element is fake.
        \*\*states
            The states of the network.
        Returns
        -------
        weighted_averages : :class:`~theano.Variable`
            Linear combinations of sequence elements with the attention
            weights.
        weights : :class:`~theano.Variable`
            The attention weights. The first dimension is batch, the second
            is time.
        """
        energies = self.compute_energies(attended, preprocessed_attended,
                                         states)
	self.myTest = energies
        self.weights = self.compute_weights(energies, attended_mask)
        self.weighted_averages = self.compute_weighted_averages(self.weights, attended)
        return self.weighted_averages, self.weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended', 'attended_mask'] +
                self.state_names)

    @application(outputs=['weighted_averages', 'weights'])
    def initial_glimpses(self, batch_size, attended):
        return [tensor.zeros((batch_size, self.attended_dim)),
                tensor.zeros((batch_size, attended.shape[0]))]

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        """Preprocess the sequence for computing attention weights.
        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The attended sequence, time is the 1-st dimension.
        """
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        return super(SequenceContentAttention2, self).get_dim(name)


class NewSoftmaxEmitter(AbstractEmitter, Initializable, Random):
    """A softmax emitter for the case of integer outputs.
    Interprets readout elements as energies corresponding to their indices.
    Parameters
    ----------
    initial_output : int or a scalar :class:`~theano.Variable`
        The initial output.
    """
    def __init__(self, initial_output=0, **kwargs):
        super(NewSoftmaxEmitter, self).__init__(**kwargs)
        self.initial_output = initial_output
        self.softmax = NDimensionalSoftmax()
        self.children = [self.softmax]
        self.name = 'newbidirectional'

    @application
    def probs(self, readouts):
        return self.softmax.apply(readouts, extra_ndim=readouts.ndim - 2)
    
    @application
    def emitProbs(self, readouts):
        probs = self.probs(readouts)
        batch_size = probs.shape[0]
        self.pvals_flat = probs.reshape((batch_size, -1))
        generated = self.theano_rng.multinomial(pvals=self.pvals_flat)
        return self.pvals_flat

    @application
    def emit(self, readouts):
        probs = self.probs(readouts)
        batch_size = probs.shape[0]
        self.pvals_flat = probs.reshape((batch_size, -1))
        generated = self.theano_rng.multinomial(pvals=self.pvals_flat)
        winning_index = generated.reshape(probs.shape).argmax(axis=-1)
        return winning_index, self.pvals_flat[0][winning_index]

    @application
    def cost(self, readouts, outputs):
        # WARNING: unfortunately this application method works
        # just fine when `readouts` and `outputs` have
        # different dimensions. Be careful!
        return self.softmax.categorical_cross_entropy(
            outputs, readouts, extra_ndim=readouts.ndim - 2)

    @application
    def initial_outputs(self, batch_size):
        return self.initial_output * tensor.ones((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(SoftmaxEmitter, self).get_dim(name)
          
# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass

class NewLookupFeedback(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in range(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup

class NewBidirectional(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]

class BidirectionalEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        
        self.lookup = LookupTable(name='embeddings')
        self.bidir = NewBidirectional(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self): 
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension.
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            # Conversion to embedding representation here. 
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        self.representation = representation
        return representation

class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    first hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)

class Decoder(Initializable):
    """Decoder of MED."""

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, theano_seed=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state.
        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Initialize the attention mechanism.
        self.attention = SequenceContentAttention2(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

	readout = Readout(
	    source_names=['states', 'feedback',
		    self.attention.take_glimpses.outputs[0]],
	    readout_dim=self.vocab_size,
	    emitter=NewSoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
	    feedback_brick=NewLookupFeedback(vocab_size, embedding_dim),
	    post_merge=InitializableFeedforwardSequence(
	        [Bias(dim=state_dim, name='maxout_bias').apply,
		Maxout(num_pieces=2, name='maxout').apply,
		Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
		       use_bias=False, name='softmax0').apply,
		Linear(input_dim=embedding_dim, name='softmax1').apply]),
	    merged_dim=state_dim)

        # Build sequence generator accordingly.
        self.sequence_generator = SequenceGenerator(
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear()),
	    cost_type='categorical_cross_entropy'
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix.
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )
        # This returns the average batch cost.
        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1] 
	  
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['test1', 'test2'])
    def getTestVar(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix.
        testVar = self.sequence_generator.getTestVar(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )
        # This returns the average batch cost.
        return testVar

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'])
    def checkRepresentation(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):
      
        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix.
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )
	
        return representation,(cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1] 

    @application
    def generate(self, source_sentence, representation, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            **kwargs)
