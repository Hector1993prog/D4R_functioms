import torch
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaForTokenClassification, BertPreTrainedModel, BertModel, BertForTokenClassification
from transformers.models.roberta.modeling_roberta import TokenClassifierOutput
from torch.nn import CrossEntropyLoss, Dropout, LSTM, Linear, LeakyReLU, GRU
from typing import  Optional, Tuple, Union

'''
This File contains costum modifications of RoBERTa and BERT architectures for Token Classification
to be used on Transfer Learning task or just normal fine-tune procedures.

'''

#################################################################################################################################################################
class RobertaTokenClassifier_With_LSTM(RobertaPreTrainedModel):
    """
    RobertaTokenClassifier_With_LSTM_We

    This class extends the `RobertaPreTrainedModel` and provides functionality for token classification tasks using a pre-trained RoBERTa model with additional layers including a bidirectional LSTM layer.

    Args:
        config (:obj:`RobertaConfig`): The configuration class for the RoBERTa model.
        negative_slope (Optional[float]): The negative slope of the LeakyReLU activation function. Default is 0.01.
        dropout_rate (Optional[float]): The dropout rate for regularization. Default is 0.1.
        num_LSTM_layers (Optional[int]): The number of LSTM layers. Default is 2.
        LSTM_hidden_size (Optional[int]): The hidden size of the LSTM layer. Default is 256.
        bidirectional (Optional[bool]): Whether the LSTM layer is bidirectional. Default is True.

    Attributes:
        num_labels (int): Number of labels for token classification.
        roberta (:obj:`RobertaModel`): The pre-trained RoBERTa model.
        roberta_dropout (:obj:`Dropout`): Dropout layer for regularization after RoBERTa.
        leaky_relu (:obj:`LeakyReLU`): LeakyReLU activation function.
        dense_1 (:obj:`Linear`): Linear layer for feature transformation.
        bidirectional (:obj:`LSTM`): Bidirectional LSTM layer.
        dropout_lstm (:obj:`Dropout`): Dropout layer for LSTM regularization.
        dense_2 (:obj:`Linear`): Second linear layer for feature transformation.
        classifier (:obj:`Linear`): Linear layer for classification.


    Returns:
        If `return_dict` is False:
        Tuple[torch.Tensor]: Tuple containing logits and optionally other hidden states.
        
        If `return_dict` is True:
        :obj:`TokenClassifierOutput`: Token classification output containing logits, loss, hidden states, and attentions.
    """
    def __init__(
            self,
            config,
            negative_slope: Optional[float]= 0.01,
            dropout_rate: Optional[float] =0.1,
            num_LSTM_layers: Optional[int]=2,
            LSTM_hidden_size: Optional[int] = 256,
            bidirectional: Optional[bool] = True):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize RoBERTa model
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Dropout for regularization
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.roberta_dropout = Dropout(p=classifier_dropout)

        # Activation function and batch normalization
        self.leaky_relu = LeakyReLU(negative_slope=negative_slope)  # type: ignore

        # Linear layer for feature transformation
        self.dense_1 = Linear(in_features=config.hidden_size,out_features= 512)

        # Bidirectional LSTM layer
        self.bidirectional = LSTM(
            input_size=512,
            hidden_size=LSTM_hidden_size,
            num_layers=num_LSTM_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Dropout for LSTM regularization
        self.dropout_lstm = Dropout(p=dropout_rate)  # type: ignore
 
        # Second Linear layer for feature transformation
        self.dense_2 = Linear(in_features=512,out_features= 256)

        # Linear layer for classification
        self.classifier = Linear(in_features=256,out_features= config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through RoBERTa model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Dropout, activation, and batch normalization
        sequence_output = self.roberta_dropout(sequence_output)
        sequence_output = self.leaky_relu(sequence_output)

        # Dense Linear feature extraction:
        dense = self.dense_1(sequence_output)

        # Bidirectional LSTM layer
        bidirectional_stack, _ = self.bidirectional(dense)

        # Applying dropout for LSTM regularization
        bidirectional_stack = self.dropout_lstm(bidirectional_stack)

        # Second Linear feature extraction
        features = self.dense_2(bidirectional_stack)

        # Linear layer for classification
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device) # type: ignore
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # type: ignore

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
 
#################################################################################################################################################################
class RobertaTokenClassifier_With_GRU(RobertaPreTrainedModel):
    """
    RobertaTokenClassifier_With_GRU

    This class extends the `RobertaPreTrainedModel` and provides functionality for token classification tasks using a pre-trained RoBERTa model with additional GRU layers.

    Args:
        config (:obj:`RobertaConfig`): The configuration class for the RoBERTa model.
        negative_slope (Optional[float]): The negative slope for the LeakyReLU activation function. Default is 0.01.
        dropout_rate (Optional[float]): The dropout rate for regularization. Default is 0.1.
        num_GRU_layers (Optional[int]): The number of GRU layers. Default is 2.
        GRU_hidden_size (Optional[int]): The hidden size of the GRU layers. Default is 256.
        bidirectional (Optional[bool]): Whether the GRU layers are bidirectional. Default is True.

    Attributes:
        num_labels (int): Number of labels for token classification.
        roberta (:obj:`RobertaModel`): The pre-trained RoBERTa model.
        roberta_dropout (:obj:`Dropout`): Dropout layer for regularization.
        leaky_relu (:obj:`LeakyReLU`): LeakyReLU activation function.
        dense_1 (:obj:`Linear`): Linear layer for feature transformation.
        bidirectional (:obj:`GRU`): Bidirectional GRU layer.
        dropout_GRU (:obj:`Dropout`): Dropout layer for GRU regularization.
        dense_2 (:obj:`Linear`): Second linear layer for feature transformation.
        classifier (:obj:`Linear`): Linear layer for classification.

ded.

    Returns:
        If `return_dict` is False:
        Tuple[torch.Tensor]: Tuple containing logits and optionally other hidden states.
        
        If `return_dict` is True:
        :obj:`TokenClassifierOutput`: Token classification output containing logits, loss, hidden states, and attentions.
    """

    def __init__(
            self,
            config,
            negative_slope: Optional[float]= 0.01,
            dropout_rate: Optional[float] =0.1,
            num_GRU_layers: Optional[int]=2,
            GRU_hidden_size: Optional[int] = 256,
            bidirectional: Optional[bool] = True
            ):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize RoBERTa model
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Dropout for regularization
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.roberta_dropout = Dropout(p=classifier_dropout)

        # Activation function and batch normalization
        self.leaky_relu = LeakyReLU(negative_slope=negative_slope)  # Adjust the negative slope as needed # type: ignore

        # Linear layer for feature transformation
        self.dense_1 = Linear(in_features=config.hidden_size,out_features= 512)

        # Bidirectional GRU layer
        self.bidirectional = GRU(
            input_size=512,
            hidden_size=GRU_hidden_size,
            num_layers=num_GRU_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Dropout for GRU regularization
        self.dropout_GRU = Dropout(p=dropout_rate)  # Adjust dropout rate as needed # type: ignore

        # Second Linear layer for feature transformation
        self.dense_2 = Linear(in_features=512,out_features= 256)

        # Linear layer for classification
        self.classifier = Linear(in_features=256,out_features= config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            attention_mask (torch.FloatTensor): Mask to avoid performing attention on padding tokens.
            token_type_ids (torch.LongTensor): Token type IDs.
            position_ids (torch.LongTensor): Positional IDs.
            head_mask (torch.FloatTensor): Mask for multi-head attention.
            inputs_embeds (torch.FloatTensor): Embedded input tokens.
            labels (torch.LongTensor): True labels for token classification.
            output_attentions (bool): Whether to return attentions weights.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return a dictionary of outputs.

        Returns:
            Union[Tuple[torch.Tensor], TokenClassifierOutput]: Tuple of model outputs or TokenClassifierOutput.

        Raises:
            NotImplementedError: If return_dict is None.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through RoBERTa model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Dropout, activation, and batch normalization
        sequence_output = self.roberta_dropout(sequence_output)
        sequence_output = self.leaky_relu(sequence_output)

        # Dense Linear feature extraction:
        dense = self.dense_1(sequence_output)

        # Bidirectional GRU layer
        bidirectional_stack, _ = self.bidirectional(dense)

        # Applying dropout for GRU regularization
        bidirectional_stack = self.dropout_GRU(bidirectional_stack)

        # Second Linear feature extraction
        features = self.dense_2(bidirectional_stack)

        # Linear layer for classification
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device) # type: ignore
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # type: ignore

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


#################################################################################################################################################################

class BertTokenClassifier_With_LSTM(BertPreTrainedModel):
    """
    BertTokenClassifier_With_LSTM_We

    This class extends the `BertPreTrainedModel` and provides functionality for token classification tasks using a pre-trained BERT model with additional layers including a bidirectional LSTM layer.

    Args:
        config (:obj:`BertConfig`): The configuration class for the BERT model.
        negative_slope (Optional[float]): The negative slope of the LeakyReLU activation function. Default is 0.01.
        dropout_rate (Optional[float]): The dropout rate for regularization. Default is 0.1.
        num_LSTM_layers (Optional[int]): The number of LSTM layers. Default is 2.
        LSTM_hidden_size (Optional[int]): The hidden size of the LSTM layer. Default is 256.
        bidirectional (Optional[bool]): Whether the LSTM layer is bidirectional. Default is True.

    Attributes:
        num_labels (int): Number of labels for token classification.
        bert (:obj:`RobertaModel`): The pre-trained BERT model.
        bert_dropout (:obj:`Dropout`): Dropout layer for regularization after BERT.
        leaky_relu (:obj:`LeakyReLU`): LeakyReLU activation function.
        dense_1 (:obj:`Linear`): Linear layer for feature transformation.
        bidirectional (:obj:`LSTM`): Bidirectional LSTM layer.
        dropout_lstm (:obj:`Dropout`): Dropout layer for LSTM regularization.
        dense_2 (:obj:`Linear`): Second linear layer for feature transformation.
        classifier (:obj:`Linear`): Linear layer for classification.


    Returns:
        If `return_dict` is False:
        Tuple[torch.Tensor]: Tuple containing logits and optionally other hidden states.
        
        If `return_dict` is True:
        :obj:`TokenClassifierOutput`: Token classification output containing logits, loss, hidden states, and attentions.
    """
    def __init__(
            self,
            config,
            negative_slope: Optional[float]= 0.01,
            dropout_rate: Optional[float] =0.1,
            num_LSTM_layers: Optional[int]=2,
            LSTM_hidden_size: Optional[int] = 256,
            bidirectional: Optional[bool] = True):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize RoBERTa model
        self.bert = BertModel(config, add_pooling_layer=False)

        # Dropout for regularization
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.bert_dropout = Dropout(p=classifier_dropout)

        # Activation function and batch normalization
        self.leaky_relu = LeakyReLU(negative_slope=negative_slope) # type: ignore

        # Linear layer for feature transformation
        self.dense_1 = Linear(in_features=config.hidden_size,out_features= 512)

        # Bidirectional LSTM layer
        self.bidirectional = LSTM(
            input_size=512,
            hidden_size=LSTM_hidden_size,
            num_layers=num_LSTM_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Dropout for LSTM regularization
        self.dropout_lstm = Dropout(p=dropout_rate)  # type: ignore

        # Second Linear layer for feature transformation
        self.dense_2 = Linear(in_features=512,out_features= 256)

        # Linear layer for classification
        self.classifier = Linear(in_features=256,out_features= config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through BERT model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Dropout, activation, and batch normalization
        sequence_output = self.bert_dropout(sequence_output)
        sequence_output = self.leaky_relu(sequence_output)

        # Dense Linear feature extraction:
        dense = self.dense_1(sequence_output)

        # Bidirectional LSTM layer
        bidirectional_stack, _ = self.bidirectional(dense)

        # Applying dropout for LSTM regularization
        bidirectional_stack = self.dropout_lstm(bidirectional_stack)

        # Second Linear feature extraction
        features = self.dense_2(bidirectional_stack)

        # Linear layer for classification
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device) # type: ignore
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # type: ignore

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
 
#################################################################################################################################################################
class BertTokenClassifier_With_GRU(BertPreTrainedModel):
    """
    BertTokenClassifier_With_GRU

    This class extends the `BertPreTrainedModel` and provides functionality for token classification tasks using a pre-trained BERT model with additional GRU layers.

    Args:
        config (:obj:`BertConfig`): The configuration class for the BERT model.
        negative_slope (Optional[float]): The negative slope for the LeakyReLU activation function. Default is 0.01.
        dropout_rate (Optional[float]): The dropout rate for regularization. Default is 0.1.
        num_GRU_layers (Optional[int]): The number of GRU layers. Default is 2.
        GRU_hidden_size (Optional[int]): The hidden size of the GRU layers. Default is 256.
        bidirectional (Optional[bool]): Whether the GRU layers are bidirectional. Default is True.

    Attributes:
        num_labels (int): Number of labels for token classification.
        bert (:obj:`BertModel`): The pre-trained BERT model.
        bert_dropout (:obj:`Dropout`): Dropout layer for regularization.
        leaky_relu (:obj:`LeakyReLU`): LeakyReLU activation function.
        dense_1 (:obj:`Linear`): Linear layer for feature transformation.
        bidirectional (:obj:`GRU`): Bidirectional GRU layer.
        dropout_GRU (:obj:`Dropout`): Dropout layer for GRU regularization.
        dense_2 (:obj:`Linear`): Second linear layer for feature transformation.
        classifier (:obj:`Linear`): Linear layer for classification.

ded.

    Returns:
        If `return_dict` is False:
        Tuple[torch.Tensor]: Tuple containing logits and optionally other hidden states.
        
        If `return_dict` is True:
        :obj:`TokenClassifierOutput`: Token classification output containing logits, loss, hidden states, and attentions.
    """

    def __init__(
            self,
            config,
            negative_slope: Optional[float]= 0.01,
            dropout_rate: Optional[float] =0.1,
            num_GRU_layers: Optional[int]=2,
            GRU_hidden_size: Optional[int] = 256,
            bidirectional: Optional[bool] = True
            ):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize BERT model
        self.bert = BertModel(config, add_pooling_layer=False)

        # Dropout for regularization
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.bert_dropout = Dropout(p=classifier_dropout)

        # Activation function and batch normalization
        self.leaky_relu = LeakyReLU(negative_slope=negative_slope)  # Adjust the negative slope as needed # type: ignore

        # Linear layer for feature transformation
        self.dense_1 = Linear(in_features=config.hidden_size,out_features= 512)

        # Bidirectional GRU layer
        self.bidirectional = GRU(
            input_size=512,
            hidden_size=GRU_hidden_size,
            num_layers=num_GRU_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Dropout for GRU regularization
        self.dropout_GRU = Dropout(p=dropout_rate)  # Adjust dropout rate as needed # type: ignore

        # Second Linear layer for feature transformation
        self.dense_2 = Linear(in_features=512,out_features= 256)

        # Linear layer for classification
        self.classifier = Linear(in_features=256,out_features= config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            attention_mask (torch.FloatTensor): Mask to avoid performing attention on padding tokens.
            token_type_ids (torch.LongTensor): Token type IDs.
            position_ids (torch.LongTensor): Positional IDs.
            head_mask (torch.FloatTensor): Mask for multi-head attention.
            inputs_embeds (torch.FloatTensor): Embedded input tokens.
            labels (torch.LongTensor): True labels for token classification.
            output_attentions (bool): Whether to return attentions weights.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return a dictionary of outputs.

        Returns:
            Union[Tuple[torch.Tensor], TokenClassifierOutput]: Tuple of model outputs or TokenClassifierOutput.

        Raises:
            NotImplementedError: If return_dict is None.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through RoBERTa model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Dropout, activation, and batch normalization
        sequence_output = self.bert_dropout(sequence_output)
        sequence_output = self.leaky_relu(sequence_output)

        # Dense Linear feature extraction:
        dense = self.dense_1(sequence_output)

        # Bidirectional GRU layer
        bidirectional_stack, _ = self.bidirectional(dense)

        # Applying dropout for GRU regularization
        bidirectional_stack = self.dropout_GRU(bidirectional_stack)

        # Second Linear feature extraction
        features = self.dense_2(bidirectional_stack)

        # Linear layer for classification
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device) # type: ignore
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # type: ignore

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#################################################################################################################################################################