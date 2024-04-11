import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size

        self.linear = nn.Linear(768, 1)

    def forward(self, state):
        value = self.linear(state)
        return value


def rl(collection, actor, optimizerA, critic, optimizerC):
    input_ids = cluster['input_ids'].to(device)
    summary = ''
    source = cluster['source']
    reward = 0
    for iter in range(episodes):
        rewards = []
        values = []
        returns = []
        actions = []
        batch = defaultdict(None)
        actor.eval()

        # input_ids = input_ids.to(device)
        # print(input_ids)

        # generate sample and calculate value
        with torch.no_grad():
            decoder_input_ids = [2]
            while len(decoder_input_ids) < max_length:
                decoder_input_ids_tensor = torch.LongTensor([decoder_input_ids]).to(device)
                logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
                dist = Categorical(logits=logits)
                action = dist.sample()
                actions.append(action)

                # TODO: Add top_k here
                # print("action = ", action)
                decoder_input_ids = decoder_input_ids + [action.item()]
                output = tokenizer.decode(decoder_input_ids + [2], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)

                batch['input_ids'] = input_ids
                batch['decoder_input_ids'] = decoder_input_ids_tensor
                batch['source'] = source
                batch['summary'] = output
                # calculate the reward of the sample
                reward_batch = env.calc_reward(batch, weights)
                reward = reward_batch['ret']
                r1 = reward_batch['R1']
                r2 = reward_batch['R2']
                r3 = reward_batch['R3']
                r4 = reward_batch['R4']
                rewards.append(reward)

                if action == 2:
                    break

            logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
            last_state = logits[0, -1]
            # print("last_state = ", last_state)
            last_value = critic(last_state)

        print(output)
        print(f"iter = {iter} reward = {reward} r1 = {r1} r2 = {r2} r3 = {r3} r4 = {r4}")

        # create calculation graph with gradient on lm_head
        final_logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
        # print("final_logits = ", final_logits)
        distributions = [Categorical(F.softmax(lgt, dim=-1)) for lgt in final_logits[0]]
        log_probs = [torch.reshape(d.log_prob(a), (-1, 1))[0] for d, a in zip(distributions, actions)]
        # print("distribution = ", distributions)
        # print("log_probs before cat = ", log_probs)

        # calculate values and returns
        ret = last_value
        for step in reversed(range(len(rewards))):
            ret = rewards[step] + gamma * ret
            returns.append(ret)
            values.append(critic(final_logits[0, step].detach()))

        # concatenate values, returns and log_probs
        # print("values before cat = ", values)
        # print("returns before cat = ", returns)
        log_probs = torch.cat(log_probs)
        # print("log_probs = ", log_probs)
        rewards = torch.FloatTensor(rewards).to(device)
        # print("rewards = ", rewards)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        # print("log_probs size = ", log_probs.size())
        # print("values size = ", values.size())
        # print("returns size = ", returns.size())

        advantages = returns.detach() - values.detach()
        # print("advantages = ", advantages)

        critic_loss = critic_loss_fct(values, returns)
        optimizerC.zero_grad()
        # critic_loss.backward(retain_graph=True)
        critic_loss.backward()

        # norm_rewards = (rewards.detach() - values.detach())
        # actor_loss = torch.mean(log_probs.mul(norm_rewards))
        actor_loss = -(log_probs * advantages.detach()).mean()

        print("actor_loss = ", actor_loss)
        print("critic_loss = ", critic_loss)

        optimizerA.zero_grad()
        actor_loss.backward()

        optimizerA.step()
        optimizerC.step()

        torch.cuda.empty_cache()

    print("final reward = ", reward)
    return reward