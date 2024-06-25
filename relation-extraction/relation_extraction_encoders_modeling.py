from torch import nn
import torch
from typing import Optional, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
from transformers.models.bert.modeling_bert import SequenceClassifierOutput

class Classifier_MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim, #It is said the dimensions are 300 in the article.
            dropout_rate,
            Gelu_aproximation,
            output_dim
            ):
        super(Classifier_MLP, self).__init__()
        self.MLP_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate=Gelu_aproximation), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate=Gelu_aproximation),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
            )
    def forward(self, x):
        pooled_input = x
        return self.MLP_block(pooled_input)

class Classifier_MLP_with_AvgPooling(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim, 
            dropout_rate,
            Gelu_aproximation,
            output_dim
            ):
        super(Classifier_MLP_with_AvgPooling, self).__init__()
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.MLP_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate=Gelu_aproximation), 
            nn.GELU(approximate=Gelu_aproximation),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
            )
    def forward(self, x):
        pooled_input = self.pooler(x.permute(0,2,1))
        pooled_input = pooled_input.squeeze(2)
        return self.MLP_block(pooled_input)

class Classifier_MLP_with_MaxPooling(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim, 
            dropout_rate,
            Gelu_aproximation,
            output_dim
            ):
        super(Classifier_MLP_with_MaxPooling, self).__init__()
        self.pooler = nn.AdaptiveMaxPool1d(1)
        self.MLP_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate=Gelu_aproximation), 
            nn.GELU(approximate=Gelu_aproximation),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
            )
    def forward(self, x):
        pooled_input = self.pooler(x.permute(0,2,1))
        pooled_input = pooled_input.squeeze(2)
        return self.MLP_block(pooled_input)
    
class PooledSimpleBertLikeForSequenceClassificationWithLSTM(BertPreTrainedModel):
    """
    Simple BERT-like model with LSTM for sequence classification.

    This model integrates BERT embeddings with a bidirectional LSTM layer
    followed by an MLP classifier for sequence classification tasks.

    Parameters:
        config (:obj:`BertConfig`): Configuration class for BERT model.
        num_positional_embeddings (int): Number of positional embeddings.
        positional_embedding_dim (int): Dimensionality of positional embeddings.
        padding_idx (Optional[int]): Index for padding tokens in positional embeddings.
        MLP_hidden_size (int): Hidden size of the MLP classifier.
        Gelu_aproximation (str): Approximation method for GELU activation function.

    Attributes:
        num_labels (int): Number of labels for sequence classification.
        config (:obj:`BertConfig`): Configuration class instance for BERT model.
        bert (:obj:`BertModel`): BERT model without pooling layer.
        positional_embedding (:obj:`nn.Embedding`): Positional embedding layer.
        Bidirectional_block (:obj:`nn.LSTM`): Bidirectional LSTM layer.
        classifier (:obj:`Classifier_MLP`): MLP classifier for final prediction.

    Methods:
        forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states,
                return_dict, sequence_positions):
            Performs forward pass through the model and returns logits or SequenceClassifierOutput.
    """
    def __init__(
            self,
            config,
            num_positional_embeddings: int = 500,
            positional_embedding_dim: int = 20,
            padding_idx: Optional[int] = 499,
            MLP_hidden_size:int = 300,
            Gelu_aproximation: str = 'none'
            ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=num_positional_embeddings,
            embedding_dim=positional_embedding_dim,
            padding_idx=padding_idx
            )

        self.Bidirectional_block = nn.LSTM(
                    input_size=self.config.hidden_size +(self.positional_embedding.embedding_dim *2),
                    hidden_size=self.config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )

    
        self.classifier = Classifier_MLP_with_AvgPooling(
            input_dim=self.config.hidden_size*2,
            hidden_dim= MLP_hidden_size,
            dropout_rate=classifier_dropout,
            Gelu_aproximation=Gelu_aproximation,
            output_dim=1
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_positions:Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        
        sep_token_id = 5
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[1][0]
        rest_tensor = outputs[0][:sep_indices+1]
        rest_positions = torch.stack([sequence_positions[0,:sep_indices+1], sequence_positions[1,:sep_indices+1]]) # type: ignore

        post_1 =self.positional_embedding(rest_positions[0])
        post_2 =self.positional_embedding(rest_positions[1])
        full_tensor = torch.cat(tensors=(rest_tensor, post_1, post_2), dim=-1)
        hidden_state_per_token, _ = self.Bidirectional_block(full_tensor)

        
        logits = self.classifier(hidden_state_per_token)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class PooledSimpleBertLikeForSequenceClassificationWithGRU(BertPreTrainedModel):
    """
    Simple BERT-like model with LSTM for sequence classification.

    This model integrates BERT embeddings with a bidirectional LSTM layer
    followed by an MLP classifier for sequence classification tasks.

    Parameters:
        config (:obj:`BertConfig`): Configuration class for BERT model.
        num_positional_embeddings (int): Number of positional embeddings.
        positional_embedding_dim (int): Dimensionality of positional embeddings.
        padding_idx (Optional[int]): Index for padding tokens in positional embeddings.
        MLP_hidden_size (int): Hidden size of the MLP classifier.
        Gelu_aproximation (str): Approximation method for GELU activation function.

    Attributes:
        num_labels (int): Number of labels for sequence classification.
        config (:obj:`BertConfig`): Configuration class instance for BERT model.
        bert (:obj:`BertModel`): BERT model without pooling layer.
        positional_embedding (:obj:`nn.Embedding`): Positional embedding layer.
        Bidirectional_block (:obj:`nn.LSTM`): Bidirectional LSTM layer.
        classifier (:obj:`Classifier_MLP`): MLP classifier for final prediction.

    Methods:
        forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states,
                return_dict, sequence_positions):
            Performs forward pass through the model and returns logits or SequenceClassifierOutput.
    """
    def __init__(
            self,
            config,
            num_positional_embeddings: int = 500,
            positional_embedding_dim: int = 20,
            padding_idx: Optional[int] = 499,
            MLP_hidden_size:int = 300,
            Gelu_aproximation: str = 'none'
            ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=num_positional_embeddings,
            embedding_dim=positional_embedding_dim,
            padding_idx=padding_idx
            )

        self.Bidirectional_block = nn.GRU(
                    input_size=self.config.hidden_size +(self.positional_embedding.embedding_dim *2),
                    hidden_size=self.config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )

    
        self.classifier = Classifier_MLP_with_AvgPooling(
            input_dim=self.config.hidden_size*2,
            hidden_dim= MLP_hidden_size,
            dropout_rate=classifier_dropout,
            Gelu_aproximation=Gelu_aproximation,
            output_dim=self.num_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_positions:Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        
        sep_token_id = 5
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[1][0]
        rest_tensor = outputs[0][:,:sep_indices+1]
        rest_positions = torch.stack([sequence_positions[:,0,:sep_indices+1], sequence_positions[:,1,:sep_indices+1]]) # type: ignore

        post_1 =self.positional_embedding(rest_positions[0])
        post_2 =self.positional_embedding(rest_positions[1])
        full_tensor = torch.cat(tensors=(rest_tensor, post_1, post_2), dim=-1)
        hidden_state_per_token, _ = self.Bidirectional_block(full_tensor)

        
        logits = self.classifier(hidden_state_per_token)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class SimpleBertLikeForSequenceClassificationWithLSTM(BertPreTrainedModel):
    """
    Simple BERT-like model with LSTM for sequence classification.

    This model integrates BERT embeddings with a bidirectional LSTM layer
    followed by an MLP classifier for sequence classification tasks.

    Parameters:
        config (:obj:`BertConfig`): Configuration class for BERT model.
        num_positional_embeddings (int): Number of positional embeddings.
        positional_embedding_dim (int): Dimensionality of positional embeddings.
        padding_idx (Optional[int]): Index for padding tokens in positional embeddings.
        MLP_hidden_size (int): Hidden size of the MLP classifier.
        Gelu_aproximation (str): Approximation method for GELU activation function.

    Attributes:
        num_labels (int): Number of labels for sequence classification.
        config (:obj:`BertConfig`): Configuration class instance for BERT model.
        bert (:obj:`BertModel`): BERT model without pooling layer.
        positional_embedding (:obj:`nn.Embedding`): Positional embedding layer.
        Bidirectional_block (:obj:`nn.LSTM`): Bidirectional LSTM layer.
        classifier (:obj:`Classifier_MLP`): MLP classifier for final prediction.

    Methods:
        forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states,
                return_dict, sequence_positions):
            Performs forward pass through the model and returns logits or SequenceClassifierOutput.
    """
    def __init__(
            self,
            config,
            num_positional_embeddings: int = 500,
            positional_embedding_dim: int = 20,
            padding_idx: Optional[int] = 499,
            MLP_hidden_size:int = 300,
            Gelu_aproximation: str = 'none'
            ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=num_positional_embeddings,
            embedding_dim=positional_embedding_dim,
            padding_idx=padding_idx
            )

        self.Bidirectional_block = nn.LSTM(
                    input_size=self.config.hidden_size +(self.positional_embedding.embedding_dim *2),
                    hidden_size=self.config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )

    
        self.classifier = Classifier_MLP(
            input_dim=self.config.hidden_size*2,
            hidden_dim= MLP_hidden_size,
            dropout_rate=classifier_dropout,
            Gelu_aproximation=Gelu_aproximation,
            output_dim=self.num_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_positions:Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        
        sep_token_id = 5
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[1][0]
        rest_tensor = outputs[0][:,:sep_indices+1]
        rest_positions = torch.stack([sequence_positions[:,0,:sep_indices+1], sequence_positions[:,1,:sep_indices+1]]) # type: ignore

        post_1 =self.positional_embedding(rest_positions[0])
        post_2 =self.positional_embedding(rest_positions[1])
        full_tensor = torch.cat(tensors=(rest_tensor, post_1, post_2), dim=-1)
        _, sequence_hidden_state = self.Bidirectional_block(full_tensor)

        final_hiddenconcat = torch.cat(tensors=(sequence_hidden_state[0][0], sequence_hidden_state[0][1]), dim=-1)
        logits = self.classifier(final_hiddenconcat)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class SimpleBertLikeForSequenceClassificationWithGRU(BertPreTrainedModel):
    """
    Simple BERT-like model with GRU for sequence classification.

    This model integrates BERT embeddings with a bidirectional LSTM layer
    followed by an MLP classifier for sequence classification tasks.

    Parameters:
        config (:obj:`BertConfig`): Configuration class for BERT model.
        num_positional_embeddings (int): Number of positional embeddings.
        positional_embedding_dim (int): Dimensionality of positional embeddings.
        padding_idx (Optional[int]): Index for padding tokens in positional embeddings.
        MLP_hidden_size (int): Hidden size of the MLP classifier.
        Gelu_aproximation (str): Approximation method for GELU activation function.

    Attributes:
        num_labels (int): Number of labels for sequence classification.
        config (:obj:`BertConfig`): Configuration class instance for BERT model.
        bert (:obj:`BertModel`): BERT model without pooling layer.
        positional_embedding (:obj:`nn.Embedding`): Positional embedding layer.
        Bidirectional_block (:obj:`nn.LSTM`): Bidirectional LSTM layer.
        classifier (:obj:`Classifier_MLP`): MLP classifier for final prediction.

    Methods:
        forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states,
                return_dict, sequence_positions):
            Performs forward pass through the model and returns logits or SequenceClassifierOutput.
    """
    def __init__(
            self,
            config,
            num_positional_embeddings: int = 500,
            positional_embedding_dim: int = 20,
            padding_idx: Optional[int] = 499,
            MLP_hidden_size:int = 300,
            Gelu_aproximation: str = 'none'
            ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=num_positional_embeddings,
            embedding_dim=positional_embedding_dim,
            padding_idx=padding_idx
            )

        self.Bidirectional_block = nn.GRU(
                    input_size=self.config.hidden_size +(self.positional_embedding.embedding_dim *2),
                    hidden_size=self.config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )

    
        self.classifier = Classifier_MLP(
            input_dim=self.config.hidden_size*2,
            hidden_dim= MLP_hidden_size,
            dropout_rate=classifier_dropout,
            Gelu_aproximation=Gelu_aproximation,
            output_dim=self.num_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_positions:Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        
        sep_token_id = 5
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[1][0]
        rest_tensor = outputs[0][:,:sep_indices+1]
        rest_positions = torch.stack([sequence_positions[:,0,:sep_indices+1], sequence_positions[:,1,:sep_indices+1]]) # type: ignore

        post_1 =self.positional_embedding(rest_positions[0])
        post_2 =self.positional_embedding(rest_positions[1])
        full_tensor = torch.cat(tensors=(rest_tensor, post_1, post_2), dim=-1)
        _, sequence_hidden_state = self.Bidirectional_block(full_tensor)

        final_hiddenconcat = torch.cat(tensors=(sequence_hidden_state[0], sequence_hidden_state[1]), dim=-1)
        logits = self.classifier(final_hiddenconcat)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )