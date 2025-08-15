# EBS Volumes

## Prompt engineering in action

Instruction:
I want a concise summary that captures the main points of the article about
- IAM
Ensure that the summary is clear and informative, focussing on key points that that are relevent to acquire deep knowledge
Include examples or analogy to explain and use less technical words.
context: I am preparing for the AWS Certified solutions architect certification and i want to prepare for details questions in the exam with multiple choice descriptive articles
Input data:
- EBS
• Network drive you attach to ONE instance only
• Linked to a specific availability zone (transfer: snapshot => restore)
• Volumes can be resized
• Make sure you choose an instance type that is EBS optimized to enjoy maximum throughput
- EBS Volume Types • EBS Volumes come in 6 types
• gp2 / gp3 (SSD): General purpose SSD volume that balances price and performance for a wide variety of workloads
• io1 / io2 Block Express: Highest-performance SSD volume for mission-critical low- latency or high-throughput workloads
• st1 (HDD): Low cost HDD volume designed for frequently accessed, throughput- intensive workloads
• sc1 (HDD): Lowest cost HDD volume designed for less frequently accessed workloads
• EBS Volumes are characterized in Size | Throughput | IOPS (I/O Ops Per Sec) • When in doubt always consult the AWS documentation – it’s good!
• Only gp2/gp3 and io1/io2 can be used as boot volumes
- EBS Snapshots
• Incremental – only backup changed blocks
• EBS backups use IO, and you shouldn’t run them while your application is handling a lot of traffic
• Snapshots will be stored in S3 (but you won’t directly see them)
• Not necessary to detach volume to do snapshot, but recommended
• Can copy snapshots across region (for DR)
• Can make Image (AMI) from Snapshot
• EBS volumes restored by snapshots need to be pre-warmed (use the Fast Snapshot Restore FSR feature or fio/dd command to read the entire volume)
- Amazon Data Lifecycle Manager
• Automate the creation, retention, and deletion of EBS snapshots and EBS-backed AMIs
• Schedule backups, cross-account snapshot copies, delete outdated backups, ...
• Uses resource tags to identify the resources (EC2 instances, EBS volumes)
• Can’t be used to manage snapshots/AMIs created outside DLM
• Can’t be used to manage instance-store backed AMIs
- Amazon Data Lifecycle Manager vs. AWS Backup
• Use Data Lifecycle Manager
• when you want to automate the creation, retention, and deletion of
EBS Snapshots
• Use AWS Backup
• to manage and monitor backups across the AWS services you use, including EBS volumes, from a single place
- EBS Encryption – Account level setting
• New Amazon EBS volumes aren’t encrypted by default
• There’s an account-level setting to encrypt automatically new EBS volumes and Snapshots
• This setting needs to be enabled on a per-region basis
- EBS Multi-Attach – io1/io2 family
• Attach the same EBS volume to multiple EC2 instances in the same AZ
• Each instance has full read & write permissions to the volume
• Use case:
• Achieve higher application availability in
clustered Linux applications (ex:Teradata)
• Applications must manage concurrent write operations
• Must use a file system that’s cluster-aware (not XFS, EXT4, etc...)
- Local EC2 Instance Store
• Physical disk attached to the physical server where your EC2 is
• Very High IOPS (because physical)
• Disks up to 7.5 TiB (can change over time), stripped to reach 60 TiB (can change over time...)
• Block Storage (just like EBS)
• Cannot be increased in size
• Risk of data loss if hardware fails
- Instance Store vs EBS
• Instance store is physically attached to the machine (ephemeral storage)
• EBS is a network drive (persistent)
• Pros:
• Better I/O performance (EBS gp2 has a max IOPS of 16000, io1 of 64000, io2
Block Express of 256000)
• Good for buffer / cache / scratch data / temporary content
• Data survives reboots
• Cons:
• On stop or termination, the instance store is lost
• You can’t resize the instance store
• Backups must be operated by the user
Output indicator:

Provide 2-3 sentence summary on each topic, subtopic and detail to showcase how I learned and my experience that covers the key learnings from my point of view.

Keep the readers curiousness from starting to end by using words like Imagine, what if, In my opinion, I would recommend, from the given choices

Do not include any emojis, Tilde, suggest using curly braces where I should include a diagram 

-- Revision slide no 320
