import logging
import os
import random

from rtp_llm.vipserver.host_reactor import HostReactor
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy


def _vipserver_disabled() -> bool:
    v = os.environ.get("RTP_DISABLE_VIPSERVER", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class DisabledVipClient:
    """Stub when not using Alibaba VIPServer: no jmenv threads or DNS noise."""

    def get_host_list_by_domain_now(self, domain: str):
        return []

    def get_host_list_by_domain(self, domain: str):
        return []

    def get_one_validate_host_now(self, domain: str):
        raise RuntimeError(
            "VIPServer is disabled (RTP_DISABLE_VIPSERVER=1); cannot resolve domain "
            f"{domain!r}. Use MODEL_SERVICE_CONFIG with use_local or enable VIPServer."
        )

    def get_one_validate_host(self, domain: str):
        return self.get_one_validate_host_now(domain)


class VipClient:

    def __init__(self, host_reactor: HostReactor):
        self.host_reactor = host_reactor
        if not self.host_reactor.started:
            self.host_reactor.start()

    def get_host_list_by_domain_now(self, domain: str):
        """
        get available host list by domain without cache, will req vipserver api right now
        :param domain: vipserver domain
        :return: host list
        """
        return self.host_reactor.get_host_list_by_domain_now(domain)

    def get_host_list_by_domain(self, domain: str):
        """
        get available host list by domain with cache
        :param domain: vipserver domain
        :return: host list
        """
        return self.host_reactor.get_host_list_by_domain(domain)

    def get_one_validate_host_now(self, domain: str):
        """
        choose one valid host randomly by domain without cache, will req vipserver api right now
        :param domain: vipserver domain
        :return: host list
        """
        return random.choice(self.get_host_list_by_domain_now(domain))

    def get_one_validate_host(self, domain: str):
        """
        choose one valid host randomly by domain with cache
        :param domain: vipserver domain
        :return: host list
        """
        return random.choice(self.get_host_list_by_domain(domain))


def _build_global_vip_client():
    if _vipserver_disabled():
        logging.info(
            "VIPServer background refresh disabled (RTP_DISABLE_VIPSERVER); "
            "safe for open-source / non-Alibaba networks."
        )
        return DisabledVipClient()
    return VipClient(HostReactor(VIPServerProxy()))


global_vip_client = _build_global_vip_client()


def get_host_list_by_domain_now(domain: str):
    """
    get available host list by domain without cache, will req vipserver api right now
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_host_list_by_domain_now(domain)


def get_host_list_by_domain(domain: str):
    """
    get available host list by domain with cache
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_host_list_by_domain(domain)


def get_one_validate_host_now(domain: str):
    """
    choose one valid host randomly by domain without cache, will req vipserver api right now
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_one_validate_host_now(domain)


def get_one_validate_host(domain: str):
    """
    choose one valid host randomly by domain with cache
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_one_validate_host(domain)
