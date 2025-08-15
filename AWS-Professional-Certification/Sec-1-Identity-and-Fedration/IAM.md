# AWS IAM – Cheat Sheet for SAP-C02 (Solutions Architect Professional)

> **Focus:** Identity & Access Management essentials for the AWS Certified Solutions Architect – Professional exam.
> Covers high-value topics like cross-account access, federation, SCPs, and policy logic.

---

##  Core IAM Concepts

* **Users** → Long-term credentials for individuals.
* **Groups** → Permission bundles assigned to multiple users.
* **Roles** → Temporary credentials, assumeable by AWS services, users, or external entities.
* **Policies** → JSON docs defining permissions.

  * AWS Managed, Customer Managed, Inline.

**Best Practice:** Enforce **least privilege** everywhere.

---

## Multi-Account Management

* **AWS Organizations** → manage multiple accounts, consolidated billing.
* **Service Control Policies (SCPs)** → define guardrails across accounts.
* **AWS Control Tower** → automated setup/governance of multi-account environments.

---

## Federation & Identity

* **IAM Identity Center (ex-SSO)** → central access to multiple accounts/apps.
* **SAML 2.0 Federation** → integrate with corporate IdP (e.g., Okta, AD).
* **Cognito** → authentication for web/mobile apps, user pools + identity pools.

---

## Temporary Credentials & STS

* **STS AssumeRole** → issue short-lived credentials.
* Used for:

  * Cross-account access.
  * Federated identities.
  * Service/application role assumption.

---

## Advanced Access Controls

* **Resource-based policies** → attached to resources (e.g., S3, KMS, Lambda).
* **Permission boundaries** → max permissions for a user/role.
* **Attribute-Based Access Control (ABAC)** → use tags + conditions for dynamic access.

Example ABAC Condition:

```json
"Condition": {
  "StringEquals": {
    "aws:ResourceTag/Project": "${aws:PrincipalTag/Project}"
  }
}
```
