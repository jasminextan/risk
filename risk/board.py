import random
from collections import namedtuple
from collections import deque
from queue import PriorityQueue
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import risk.definitions

Territory = namedtuple('Territory', ['territory_id', 'player_id', 'armies'])
Move = namedtuple('Attack', ['from_territory_id',
                             'from_armies', 'to_territory_id',
                             'to_player_id', 'to_armies'])


class Board(object):
    """
    The Board object keeps track of all armies situated on the Risk
    world map. Through the definitions it knows the locations of and
    connections between all territories. It handles ownership, attacks
    and movements of armies.

    Args:
        data (list): a sorted list of tuples describing the state of the
            board, each containing three values:
            - tid (int): the territory id of a territory,
            - pid (int): the player id of the owner of the territory,
            - n_armies (int): the number of armies on the territory.
            The list is sorted by the tid, and should be complete.
    """

    def __init__(self, data):
        self.data = data

    @classmethod
    def create(cls, n_players):
        """
        Args:
            n_players (int): Number of players.
        Returns:
            Board: A board with territories randomly allocated to the players.
        """
        allocation = (list(range(n_players)) * 42)[0:42]
        random.shuffle(allocation)
        return cls([Territory(territory_id=tid,
                    player_id=pid, armies=1) for tid, pid in
                    enumerate(allocation)])

    # ====================== #
    # == Neighbor Methods == #
    # ====================== #

    def neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory.
        Args:
            territory_id (int): ID of the territory to find neighbors of.

        Returns:
            generator: Generator of Territories.
        """
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if t.territory_id in neighbor_ids)

    def hostile_neighbors(self, territory_id):
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if
                (t.player_id != player_id and t.territory_id in neighbor_ids))

    def friendly_neighbors(self, territory_id):
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if
                (t.player_id == player_id and t.territory_id in neighbor_ids))

    # ================== #
    # == Path Methods == #
    # ================== #

    def is_valid_path(self, path):
        if not len(set(path)) == len(path):
            return False

        def go(xs):
            if len(xs) <= 1:
                return True
            neighbors = list(self.neighbors(xs[0]))
            neighbors = [neighbor.territory_id for neighbor in neighbors]
            if xs[1] in neighbors:
                return self.is_valid_path(xs[1:])
            else:
                return False
        return go(path)

    def is_valid_attack_path(self, path):
        if not len(set(path)) == len(path) or len(path) < 2:
            return False

        def go(xs, origin):
            if len(xs) <= 1:
                return True
            print("list(self.neighbors(xs[0]))=", list(self.neighbors(xs[0])))
            neighbors = list(self.neighbors(xs[0]))
            neighbors = [neighbor.territory_id for neighbor
                         in neighbors if neighbor.player_id != origin]
            print("neighbors=", neighbors)
            print("origin=", origin)
            if xs[1] in neighbors:
                return go(xs[1:], origin)
            else:
                return False
        hostile_neighbors = list(self.hostile_neighbors(path[0]))
        hostile_neighbors = [neighbor.territory_id for neighbor
                             in hostile_neighbors]
        if path[1] in hostile_neighbors:
            player0 = self.owner(path[0])
            return go(path[1:], player0)
        else:
            return False

    def cost_of_attack_path(self, path):
        if self.is_valid_attack_path(path):
            def go(xs, cost=0):
                armies_list = [t.armies for t in self.data
                               if t.territory_id == xs[0]]
                cost += armies_list[0]
                if len(xs) == 1:
                    return cost
                else:
                    return go(xs[1:], cost)
            return go(path[1:])

    def shortest_path(self, source, target):
        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = []
        visited.append(source)

        while len(queue) != 0:
            print("queue=", queue)
            current_territory = queue.popleft()
            if current_territory == target:
                return dictionary[current_territory]
            else:
                neighbors = list(self.neighbors(current_territory))
                neighbors = [neighbor.territory_id for neighbor
                             in neighbors if
                             neighbor.territory_id not in visited]
                print("neighbors=", neighbors)
                for territory in neighbors:
                    if territory not in queue:
                        copy_path = copy.copy(dictionary[current_territory])
                        copy_path.append(territory)
                        dictionary[territory] = copy_path
                        print("dictionary[territory]=", dictionary[territory])
                        queue.append(territory)
                visited.append(current_territory)

    def can_fortify(self, source, target):
        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = []
        visited.append(source)

        while len(queue) != 0:
            print("queue=", queue)
            current_territory = queue.popleft()
            if current_territory == target:
                return True
            else:
                neighbors = list(self.friendly_neighbors(current_territory))
                neighbors = [neighbor.territory_id for neighbor
                             in neighbors if
                             neighbor.territory_id not in visited]
                print("neighbors=", neighbors)
                for territory in neighbors:
                    if territory not in queue:
                        copy_path = copy.copy(dictionary[current_territory])
                        copy_path.append(territory)
                        dictionary[territory] = copy_path
                        queue.append(territory)
                visited.append(current_territory)
        return False

    def cheapest_attack_path(self, source, target):
        '''
        Args:
            source (int): territory_id of source node
            target (int): territory_id of target node

        Returns:
            valid attack path; if no path exists, then it returns None instead
        '''
        if source == target:
            return None
        dictionary = {}
        dictionary[source] = [source]
        queue = PriorityQueue()
        queue.put((0, source))
        visited = []
        visited.append(source)
        player0 = self.owner(source)

        while queue.qsize() != 0:
            current_territory = queue.get()
            if current_territory[1] == target:
                return dictionary[current_territory[1]]
            else:
                neighbors = list(self.neighbors(current_territory[1]))
                neighbors = [neighbor.territory_id for neighbor in neighbors
                             if neighbor.territory_id not in visited and
                             neighbor.player_id != player0]
                for territory in neighbors:
                    copy_path = copy.copy(dictionary[current_territory[1]])
                    copy_path.append(territory)
                    priority = current_territory[0] + self.armies(territory)
                    if territory not in [item[1] for item in queue.queue]:
                        dictionary[territory] = copy_path
                        queue.put((priority, territory))
                    else:
                        already_in = [item for item in
                                      queue.queue if
                                      item[1] == territory]
                        if priority < already_in[0][0]:
                            queue.put((priority, territory))
                            queue.queue.remove(already_in)
                visited.append(current_territory[1])
        return None

    def can_attack(self, source, target):
        '''
        Args:
            source (int): territory_id of source node
            target (int): territory_id of target node

        Returns:
        '''
        if self.cheapest_attack_path(source, target):
            return True
        else:
            return False

    # ======================= #
    # == Continent Methods == #
    # ======================= #

    def continent(self, continent_id):
        """
        Create a generator of all territories that belong to a given continent.
        Args:
            continent_id (int): ID of the continent.

        Returns:
            generator: Generator of Territories.
        """
        return (t for t in self.data if t.territory_id
                in risk.definitions.continent_territories[continent_id])

    def n_continents(self, player_id):
        """
        Calculate the total number of continents owned by a player.
        Args:
            player_id (int): ID of the player.
        Returns:
            int: Number of continents owned by the player.
        """
        return len([continent_id for continent_id in
                    range(6) if self.owns_continent(player_id, continent_id)])

    def owns_continent(self, player_id, continent_id):
        """
        Check if a player owns a continent.
        Args:
            player_id (int): ID of the player.
            continent_id (int): ID of the continent.
        Returns:
            bool: True if the player owns all of the continent's territories.
        """
        return all((t.player_id == player_id
                    for t in self.continent(continent_id)))

    def continent_owner(self, continent_id):
        """
        Find the owner of all territories in a continent. If the continent
        is owned by various players, return None.
        Args:
            continent_id (int): ID of the continent.
        Returns:
            int/None: Player_id if a player owns all territories, else None.
        """
        pids = set([t.player_id for t in self.continent(continent_id)])
        if len(pids) == 1:
            return pids.pop()
        return None

    def continent_fraction(self, continent_id, player_id):
        """
        Compute the fraction of a continent a player owns.
        Args:
            continent_id (int): ID of the continent.
            player_id (int): ID of the player.

        Returns:
            float: The fraction of the continent owned by the player.
        """
        c_data = list(self.continent(continent_id))
        p_data = [t for t in c_data if t.player_id == player_id]
        return float(len(p_data)) / len(c_data)

    def num_foreign_continent_territories(self, continent_id, player_id):
        """
        Args:
            continent_id (int): ID of the continent.
            player_id (int): ID of the player.
        """
        return sum(1 if t.player_id !=
                   player_id else 0 for t in self.continent(continent_id))

    # ==================== #
    # == Action Methods == #
    # ==================== #

    def reinforcements(self, player_id):
        """
        Calculate the number of reinforcements a player is entitled to.
        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of reinforcement armies that the player is entitled to.
        """
        base_reinforcements = max(3, int(self.n_territories(player_id) / 3))
        bonus_reinforcements = 0
        for continent_id, bonus in risk.definitions.continent_bonuses.items():
            if self.continent_owner(continent_id) == player_id:
                bonus_reinforcements += bonus
        return base_reinforcements + bonus_reinforcements

    def possible_attacks(self, player_id):
        """
        Assemble a list of all possible attacks for the players.

        Args:
            player_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.territory_id, from_t.armies,
                to_t.territory_id, to_t.player_id, to_t.armies)
                for from_t in self.mobile(player_id)
                for to_t in self.hostile_neighbors(from_t.territory_id)]

    def possible_fortifications(self, player_id):
        return [Move(from_t.territory_id, from_t.armies,
                to_t.territory_id, to_t.player_id, to_t.armies)
                for from_t in self.mobile(player_id)
                for to_t in self.friendly_neighbors(from_t.territory_id)]

    def fortify(self, from_territory, to_territory, n_armies):
        if n_armies < 0 or self.armies(from_territory) <= n_armies:
            raise ValueError('Board: Cannot move')
        if to_territory not in [t.territory_id
                                for t in
                                self.friendly_neighbors(from_territory)]:
            raise ValueError('Board: Cannot fortify')
        self.add_armies(from_territory, -n_armies)
        self.add_armies(to_territory, +n_armies)

    def attack(self, from_territory, to_territory, attackers):
        if attackers < 1 or self.armies(from_territory) <= attackers:
            raise ValueError('Board: Cannot attack')
        if to_territory not in [tid for (tid, _, _)
                                in self.hostile_neighbors(from_territory)]:
            raise ValueError('Board: Cannot attack')
        defenders = self.armies(to_territory)
        def_wins, att_wins = self.fight(attackers, defenders)
        if self.armies(to_territory) == att_wins:
            self.add_armies(from_territory, -attackers)
            self.set_armies(to_territory, attackers - def_wins)
            self.set_owner(to_territory, self.owner(from_territory))
            return True
        else:
            self.add_armies(from_territory, -def_wins)
            self.add_armies(to_territory, -att_wins)
            return False

    # ====================== #
    # == Plotting Methods == #
    # ====================== #

    def plot_board(self, path=None, plot_graph=False, filename=None):
        # im = plt.imread(os.getcwd() + '/img/risk.png')
        dpi = 96
        img_width = 800
        fig, ax = plt.subplots(figsize=(img_width/dpi, 300/dpi), dpi=dpi)
        # _ = plt.imshow(im)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        def plot_path(xs):
            if not self.is_valid_path(xs):
                print('WARNING: not a valid path')
            coor = risk.definitions.territory_locations[xs[0]]
            verts = [(coor[0] * 1.2, coor[1] * 1.22 + 25)]
            codes = [Path.MOVETO]
            for i, x in enumerate(xs[1:]):
                if (xs[i] == 19 and xs[i + 1]
                        == 1) or (xs[i] == 1 and xs[i + 1] == 19):
                    coor = risk.definitions.territory_locations[x]
                    # verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    verts.append((1000, -200))
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.CURVE3)
                    codes.append(Path.CURVE3)
                else:
                    coor = risk.definitions.territory_locations[x]
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.LINETO)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)

        if path is not None:
            plot_path(path)

        if plot_graph:
            for t in risk.definitions.territory_neighbors:
                path = []
                for n in risk.definitions.territory_neighbors[t]:
                    path.append(t)
                    path.append(n)
                plot_path(path)

        for t in self.data:
            self.plot_single(t.territory_id, t.player_id, t.armies)

        if not filename:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')

    @staticmethod
    def plot_single(territory_id, player_id, armies):
        """
        Plot a single army dot.
        Args:
            territory_id (int): the id of the territory to plot,
            player_id (int): the player id of the owner,
            armies (int): the number of armies.
        """
        coor = risk.definitions.territory_locations[territory_id]
        plt.scatter(
            [coor[0]*1.2],
            [coor[1]*1.22],
            s=400,
            c=risk.definitions.player_colors[player_id],
            zorder=2
            )
        plt.text(
            coor[0]*1.2,
            coor[1]*1.22 + 15,
            s=str(armies),
            color='black' if
            risk.definitions.player_colors[player_id]
            in ['yellow', 'pink'] else 'white',
            ha='center',
            size=15
            )

    # ==================== #
    # == Combat Methods == #
    # ==================== #

    @classmethod
    def fight(cls, attackers, defenders):
        n_attack_dices = min(attackers, 3)
        n_defend_dices = min(defenders, 2)
        attack_dices = sorted([cls.throw_dice()
                              for _ in range(n_attack_dices)], reverse=True)
        defend_dices = sorted([cls.throw_dice()
                              for _ in range(n_defend_dices)], reverse=True)
        wins = [att_d > def_d for att_d, def_d in
                zip(attack_dices, defend_dices)]
        return len([w for w in wins if w is False]),
        len([w for w in wins if w is True])

    @staticmethod
    def throw_dice():
        """
        Throw a dice.

        Returns:
            int: random int in [1, 6]. """
        return random.randint(1, 6)

    # ======================= #
    # == Territory Methods == #
    # ======================= #

    def owner(self, territory_id):
        """
        Get the owner of the territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            int: Player_id that owns the territory.
        """
        return self.data[territory_id].player_id

    def armies(self, territory_id):
        """
        Get the number of armies on the territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            int: Number of armies in the territory.
        """
        return self.data[territory_id].armies

    def set_owner(self, territory_id, player_id):
        """
        Set the owner of the territory.

        Args:
            territory_id (int): ID of the territory.
            player_id (int): ID of the player.
        """
        self.data[territory_id] = Territory(territory_id,
                                            player_id,
                                            self.armies(territory_id))

    def set_armies(self, territory_id, n):
        """
        Set the number of armies on the territory.

        Args:
            territory_id (int): ID of the territory.
            n (int): Number of armies on the territory.

        Raises:
            ValueError if n < 1.
        """
        if n < 1:
            raise ValueError('Board: cannot set the number')
        self.data[territory_id] = Territory(territory_id, self.owner(
            territory_id), n)

    def add_armies(self, territory_id, n):
        """
        Add (or remove) armies to/from the territory.

        Args:
            territory_id (int): ID of the territory.
            n (int): Number of armies to add to the territory.

        Raises:
            ValueError if the resulting number of armies is <1.
        """
        self.set_armies(territory_id, self.armies(territory_id) + n)

    def n_armies(self, player_id):
        """
        Count the total number of armies owned by a player.

        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of armies owned by the player.
        """
        return sum((t.armies for t in self.data if t.player_id == player_id))

    def n_territories(self, player_id):
        """
        Count the total number of territories owned by a player.

        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of territories owned by the player.
        """
        return len([None for t in self.data if t.player_id == player_id])

    def territories_of(self, player_id):
        """
        Return a set of all territories owned by the player.

        Args:
            player_id (int): ID of the player.

        Returns:
            list: List of all territory IDs owner by the player.
        """
        return [t.territory_id for t in self.data if t.player_id == player_id]

    def mobile(self, player_id):
        return (t for t in self.data if (t.player_id ==
                player_id and t.armies > 1))
